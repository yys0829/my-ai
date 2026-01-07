import streamlit as st
import os, sys, json, tempfile, uuid

# --- 1. 环境补丁 ---
venv_pkg = os.path.join(os.getcwd(), "venv", "Lib", "site-packages")
if venv_pkg not in sys.path: sys.path.insert(0, venv_pkg)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except Exception as e:
    st.error(f"组件缺失：{e}"); st.stop()

# --- 2. 基础配置 ---
DB_PREFIX = "db_"
HISTORY_FILE = "all_chats_v3.json"

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

def load_all_chats():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def save_all_chats(chats):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

# --- 3. 页面设置 ---
st.set_page_config(page_title="DeepSeek 集团全能智库", layout="wide")
# --- 权限校验功能 ---
def check_password():
    """如果返回 True，则显示主界面；如果返回 False，则停留在登录页"""
    def password_entered():
        # 这里设置你的访问密码
        if st.session_state["password"] == "6688": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 登录成功后删除临时存储的密码
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # 还没登录过，显示输入框
        st.title("🔐 私有智库访问授权")
        st.text_input("请输入访问授权码", type="password", on_change=password_entered, key="password")
        st.info("提示：此为个人私有办公智库，仅限授权使用。")
        return False
    elif not st.session_state["password_correct"]:
        # 密码输错了
        st.title("🔐 私有智库访问授权")
        st.text_input("授权码错误，请重新输入", type="password", on_change=password_entered, key="password")
        st.error("❌ 授权失败")
        return False
    else:
        # 密码正确
        return True

# --- 逻辑控制 ---
if not check_password():
    st.stop()  # 如果没登录成功，直接切断后续所有代码的执行

# --- 后面就是你原来的代码了 (st.sidebar 等等) ---
if "all_chats" not in st.session_state: st.session_state.all_chats = load_all_chats()
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None

# --- 4. 侧边栏 ---
with st.sidebar:
    st.title("📂 智库管理中心")
    
    # A. 跨库检索开关
    st.subheader("🛠️ 检索模式")
    multi_db_mode = st.toggle("🌐 开启全库联合检索", value=False, help="开启后将搜索所有分类库，适合做跨逻辑对比。")
    
    # B. 知识库维护
    with st.expander("✨ 知识库维护 (上传/新建)"):
        existing_dirs = [d.replace(DB_PREFIX, "") for d in os.listdir(".") if os.path.isdir(d) and d.startswith(DB_PREFIX)]
        op_mode = st.radio("模式", ["现有分类", "新分类"], horizontal=True)
        target_cat = st.selectbox("选择分类", existing_dirs) if op_mode == "现有分类" else st.text_input("新分类名称")
        uploaded_files = st.file_uploader("上传文件", accept_multiple_files=True)
        
        if st.button("🚀 运行构建"):
            if not target_cat or not uploaded_files: st.warning("请完善信息")
            else:
                with st.spinner("处理中..."):
                    all_docs = []
                    for f in uploaded_files:
                        ext = os.path.splitext(f.name)[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(f.getvalue()); tmp_path = tmp.name
                        try:
                            if ext == ".pdf": loader = PyPDFLoader(tmp_path)
                            elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
                            else: loader = TextLoader(tmp_path, encoding="utf-8")
                            all_docs.extend([d for d in loader.load() if d.page_content.strip()])
                        finally: os.unlink(tmp_path)
                    
                    if all_docs:
                        splits = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(all_docs)
                        Chroma.from_documents(documents=splits, embedding=get_embedding_model(), persist_directory=f"./{DB_PREFIX}{target_cat}")
                        st.success("构建成功！"); st.rerun()

    st.divider()
    
    # C. 检索范围（非联合模式下生效）
    st.subheader("🔍 问答检索范围")
    all_cats = [d.replace(DB_PREFIX, "") for d in os.listdir(".") if os.path.isdir(d) and d.startswith(DB_PREFIX)]
    selected_cat = st.selectbox("当前提问基于：", all_cats if all_cats else ["默认"], disabled=multi_db_mode)

    st.divider()
    
    # D. 历史话题
    st.subheader("🕙 历史话题")
    if st.button("➕ 开启新对话"): st.session_state.current_chat_id = None; st.rerun()
    for cid, cdata in reversed(list(st.session_state.all_chats.items())):
        if st.button(f"💬 {cdata['title']}", key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid; st.rerun()

    st.divider()
    api_key = st.text_input("DeepSeek API Key", type="password")

# --- 5. 主界面与问答逻辑 ---
st.markdown(f"### 🎯 模式：{'全库联合检索' if multi_db_mode else f'单库检索({selected_cat})'}")

if st.session_state.current_chat_id:
    for m in st.session_state.all_chats[st.session_state.current_chat_id]["messages"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
else:
    st.info("请在下方输入问题。开启'全库联合检索'可同时对比多个分类文件。")

if prompt := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"): st.markdown(prompt)
    if not st.session_state.current_chat_id:
        cid = str(uuid.uuid4()); st.session_state.current_chat_id = cid
        st.session_state.all_chats[cid] = {"title": prompt[:12], "messages": []}

    if not api_key: st.error("请配置 API Key")
    else:
        with st.chat_message("assistant"):
            with st.spinner("正在跨库检索资料..."):
                try:
                    # --- 核心逻辑：多库联合加载 ---
                    combined_context = ""
                    search_list = all_cats if multi_db_mode else [selected_cat]
                    
                    for cat in search_list:
                        db_p = f"./{DB_PREFIX}{cat}"
                        if os.path.exists(db_p):
                            vdb = Chroma(persist_directory=db_p, embedding_function=get_embedding_model())
                            docs = vdb.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(prompt)
                            combined_context += f"\n\n--- 来自【{cat}】的参考资料 ---\n"
                            combined_context += "\n".join([d.page_content for d in docs])
                    
                    if not combined_context.strip():
                        response = "未能在任何知识库中找到相关资料。"
                    else:
                        llm = ChatOpenAI(model='deepseek-chat', openai_api_key=api_key, openai_api_base="https://api.deepseek.com", temperature=0.1)
                        prompt_tmpl = ChatPromptTemplate.from_template("你是一个企业助手。请根据以下资料回答。如果资料来自不同库，请对比分析。\n资料：{context}\n问题：{question}")
                        chain = ({"context": lambda x: combined_context, "question": RunnablePassthrough()} | prompt_tmpl | llm | StrOutputParser())
                        response = chain.invoke(prompt)
                    
                    st.markdown(response)
                    st.session_state.all_chats[st.session_state.current_chat_id]["messages"].append({"role": "user", "content": prompt})
                    st.session_state.all_chats[st.session_state.current_chat_id]["messages"].append({"role": "assistant", "content": response})
                    save_all_chats(st.session_state.all_chats)
                except Exception as e: st.error(f"出错：{e}")