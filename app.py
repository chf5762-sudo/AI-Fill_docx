import streamlit as st
import streamlit_authenticator as stauth
import json
import os
import base64
import re
import requests
from io import BytesIO
from docx import Document
import PIL.Image 
import yaml
from yaml.loader import SafeLoader
# 确保已安装所需的库
from openai import OpenAI, APIConnectionError, AuthenticationError, BadRequestError
import anthropic
import google.generativeai as genai

# ==============================================================================
#                      智能文档填充工具的核心逻辑 (封装在函数中)
# ==============================================================================

def run_app():
    # ⚠️ 注意: 假设 prompt_library.py 不存在，此处简化处理，移除了导入，使用硬编码的默认模板
    PROMPT_LIBRARY_AVAILABLE = False
    DEFAULT_TEMPLATES = {}
    GLOBAL_INSTRUCTIONS = ""

    # 配置变量，确保在 Streamlit Cloud 部署时功能正常
    CONFIG_FILE = "api_config.json" 
    
    API_TYPES = {
        "openai_official": {
            "name": "OpenAI 官方",
            "needs_url": False,
            "default_url": "https://api.openai.com/v1",
            "default_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        },
        "claude_official": {
            "name": "Claude 官方",
            "needs_url": False,
            "default_url": None,
            "default_models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
        },
        "gemini_official": {
            "name": "Gemini 官方",
            "needs_url": False,
            "default_url": None,
            "default_models": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
        },
        "openai_custom": {
            "name": "OpenAI 自定义",
            "needs_url": True,
            "default_url": "",
            "default_models": []
        },
        "claude_custom": {
            "name": "Claude 自定义",
            "needs_url": True,
            "default_url": "",
            "default_models": []
        },
        "gemini_custom": {
            "name": "Gemini 自定义",
            "needs_url": True,
            "default_url": "",
            "default_models": []
        }
    }
    
    # ========== 样式 (与您的原始代码一致) ==========
    st.markdown("""
    <style>
        html, body, [class*="css"] { font-size: 16px; }
        .stMarkdown, .stText, p, div, span, label { font-size: 1.1rem !important; }
        .stButton button { font-size: 1.1rem !important; }
        .stTextInput input, .stTextArea textarea, .stSelectbox select { font-size: 1.1rem !important; }
        .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label { font-size: 1.1rem !important; }
        .stAlert, .stInfo, .stWarning, .stSuccess, .stError { font-size: 1.1rem !important; }
        .stCodeBlock, code { font-size: 1rem !important; }
        .streamlit-expanderHeader { font-size: 1.2rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 1.2rem !important; }
        .main-header { font-size: 2.5rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem; }
        .sub-header { font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem; }
        .model-info { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 0.5rem; display: flex; justify-content: space-between; align-items: center; font-size: 1.1rem; }
        .replace-preview { background: #f3f4f6; border-left: 4px solid #3b82f6; padding: 1rem; margin: 0.5rem 0; border-radius: 0.25rem; font-size: 1.1rem; }
        .success-box { background: #d1fae5; border: 2px solid #10b981; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; font-size: 1.1rem; }
        .error-detail { background: #fee2e2; border: 2px solid #ef4444; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; font-family: monospace; font-size: 1rem; max-height: 300px; overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

    # ========== 配置管理 (简化 Streamlit Cloud 上的文件读写，使用默认配置) ==========
    def load_config():
        """加载配置文件 - 在 Streamlit Cloud 上使用默认值"""
        return {
            'api_type': 'gemini_custom',
            'api_key': '',
            'base_url': '',
            'model_name': '',
            'model_list': [],
            'prompt_settings': {
                'global_prompt': '',
                'templates': DEFAULT_TEMPLATES.copy() if PROMPT_LIBRARY_AVAILABLE else {},
            }
        }

    def save_config():
        """保存配置 - 在 Streamlit Cloud 上仅更新 session_state"""
        # 实际部署时无法持久化保存到文件，此处仅为结构保留
        pass 

    # ========== URL 处理工具 (与您的原始代码一致) ==========
    def get_clean_base_url(url):
        """清洗并标准化 Base URL"""
        if not url:
            return ""
        clean = url.strip().rstrip('/')
        if clean.endswith('/chat/completions'):
            clean = clean.replace('/chat/completions', '')
        if clean.endswith('/models'):
            clean = clean.replace('/models', '')
        # 对于 OpenAI 兼容接口，通常需要 /v1 后缀
        if not clean.endswith('/v1') and "custom" in st.session_state.api_type: 
            clean += '/v1'
        return clean

    # ========== 模型获取功能 (与您的原始代码一致) ==========
    def fetch_models_list(api_type, api_key, base_url):
        """获取模型列表（统一接口）"""
        
        # 官方 OpenAI
        if api_type == "openai_official":
            try:
                client = OpenAI(api_key=api_key, timeout=10)
                models = client.models.list()
                return [m.id for m in models.data if 'gpt' in m.id.lower()], None
            except Exception as e:
                return None, f"OpenAI 官方连接失败: {str(e)}"
        
        # 官方 Claude
        elif api_type == "claude_official":
            return API_TYPES["claude_official"]["default_models"], None
        
        # 官方 Gemini
        elif api_type == "gemini_official":
            try:
                genai.configure(api_key=api_key)
                models = genai.list_models()
                model_names = [m.name.replace('models/', '') for m in models if 'generateContent' in m.supported_generation_methods]
                return model_names if model_names else API_TYPES["gemini_official"]["default_models"], None
            except Exception as e:
                return None, f"Gemini 官方连接失败: {str(e)}"
        
        # 自定义 API（OpenAI 兼容格式）
        elif api_type in ["openai_custom", "claude_custom", "gemini_custom"]:
            if not base_url:
                return None, "请填写 Base URL"
            
            clean_url = get_clean_base_url(base_url)
            models_url = f"{clean_url.replace('/v1', '')}/models" # 有些反代不支持 /v1/models
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key if api_key else 'sk-dummy'}"
            }
            
            try:
                response = requests.get(models_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and isinstance(data['data'], list):
                        return [m['id'] for m in data['data']], None
                    return None, "返回格式异常"
                else:
                    return None, f"HTTP {response.status_code}"
            except Exception as e:
                return None, f"连接失败: {str(e)}"
        
        return None, "未知 API 类型"

    # ========== API 测试功能 (与您的原始代码一致) ==========
    def test_api_connection(api_type, api_key, base_url, model_name):
        """测试 API 连接"""
        test_prompt = "请回复：OK"
        
        try:
            response, error = call_ai_api(test_prompt, api_type, api_key, base_url, model_name)
            if error:
                return False, error
            if response and len(response) > 0:
                return True, "连接成功！"
            return False, "返回内容为空"
        except Exception as e:
            return False, str(e)

    # ========== 核心 API 调用 (与您的原始代码一致) ==========
    def call_ai_api(prompt, api_type=None, api_key=None, base_url=None, model_name=None, image_data=None, custom_prompt=None):
        """统一的 AI 调用接口"""
        
        # 获取配置
        if api_type is None:
            api_type = st.session_state.get('api_type', 'gemini_custom')
        if api_key is None:
            api_key = st.session_state.get('api_key', '')
        if base_url is None:
            base_url = st.session_state.get('base_url', '')
        if model_name is None:
            model_name = st.session_state.get('model_name', '')
        
        # 增强提示词
        enhanced_prompt = get_enhanced_prompt(prompt, custom_prompt)
        
        try:
            # ========== OpenAI 官方 & 自定义 API (兼容) ==========
            if api_type == "openai_official" or api_type in ["openai_custom", "claude_custom", "gemini_custom"]:
                
                if "custom" in api_type and not base_url:
                    return None, "请配置 Base URL"
                    
                client = OpenAI(
                    api_key=api_key if api_key else "sk-dummy",
                    base_url=get_clean_base_url(base_url) if "custom" in api_type else None,
                    timeout=60.0,
                    max_retries=1,
                    default_headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    }
                )
                
                content_list = [{"type": "text", "text": enhanced_prompt}]
                
                if image_data and ('gpt-4' in model_name or 'custom' in api_type or 'gemini' in model_name):
                    content_list.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                
                messages = [{"role": "user", "content": content_list}]
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content, None
            
            # ========== Claude 官方 ==========
            elif api_type == "claude_official":
                if not api_key:
                    return None, "请配置 Claude API Key"
                
                client = anthropic.Anthropic(api_key=api_key)
                
                if image_data:
                    content = [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                        {"type": "text", "text": enhanced_prompt}
                    ]
                else:
                    content = [{"type": "text", "text": enhanced_prompt}]
                
                message = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": content}]
                )
                return message.content[0].text, None
            
            # ========== Gemini 官方 ==========
            elif api_type == "gemini_official":
                if not api_key:
                    return None, "请配置 Gemini API Key"
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                contents = [enhanced_prompt]
                if image_data:
                    # 必须使用 PIL.Image.open 从 BytesIO 创建图像对象
                    img = PIL.Image.open(BytesIO(base64.b64decode(image_data)))
                    contents.insert(0, img)

                response = model.generate_content(contents)
                
                return response.text, None
            
            else:
                return None, f"未知的 API 类型: {api_type}"
        
        except AuthenticationError:
            return None, "❌ 认证失败，请检查 API Key"
        except APIConnectionError as e:
            return None, f"❌ 连接失败: {str(e)}"
        except BadRequestError as e:
            # 捕捉模型不支持图片等具体错误
            return None, f"❌ 请求错误: {str(e)}"
        except Exception as e:
            return None, f"❌ 未知错误: {str(e)}"

    # ========== JSON 处理工具 (与您的原始代码一致) ==========
    def clean_json_response(response_text):
        """清理 AI 返回的 JSON 响应"""
        if not response_text:
            return ""
        
        text = response_text.strip()
        
        # 提取 ```json ``` 代码块
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        
        # 正则提取 JSON 对象
        match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return text

    def parse_json_safely(response_text, context=""):
        """安全解析 JSON"""
        try:
            cleaned = clean_json_response(response_text)
            result = json.loads(cleaned)
            
            filtered = {}
            for k, v in result.items():
                if v:
                    val = v.get('value', '') if isinstance(v, dict) else str(v)
                    if val.strip():
                        filtered[k] = val.strip()
            
            return filtered, None
        except Exception as e:
            error_details = {
                'error': f"JSON 解析失败 ({context}): {str(e)}",
                'original_response': response_text[:500],
                'cleaned_response': clean_json_response(response_text)[:500]
            }
            return None, error_details

    # ========== 提示词增强 (与您的原始代码一致) ==========
    def get_enhanced_prompt(base_prompt, custom_prompt=None):
        """增强提示词"""
        if custom_prompt and custom_prompt.strip():
            return f"{custom_prompt}\n\n{base_prompt}"
        
        if not PROMPT_LIBRARY_AVAILABLE:
            return base_prompt
        
        prompt_settings = st.session_state.get('prompt_settings', {})
        enhanced = base_prompt
        
        templates = prompt_settings.get('templates', {})
        for name, template_config in templates.items():
            if template_config.get('enabled', False):
                if t_prompt := template_config.get('prompt', '').strip():
                    enhanced = f"【文档类型: {name}】\n{t_prompt}\n\n{enhanced}"
                break
        
        if g_prompt := prompt_settings.get('global_prompt', '').strip():
            enhanced = f"【全局指令】\n{g_prompt}\n\n{enhanced}"
        
        return enhanced

    # ========== 业务逻辑函数 (与您的原始代码一致) ==========
    def extract_customer_info_from_text(text, image_data=None, custom_prompt=None):
        """从用户输入中提取信息"""
        prompt = f"""
你是信息提取专家，从混乱的文本中全面识别信息。

【用户输入】
{text if text else '无文本输入'}

【识别规则】
请提取以下所有能找到的信息：
1. 公司名称、联系人姓名、职位/部门
2. 联系电话、手机号码、固定电话、传真号码
3. 电子邮箱、详细地址、邮政编码、公司网址
4. 统一社会信用代码、法人代表、开户行信息、银行账号
5. 任何其他看起来重要的信息

【输出格式】
只返回纯 JSON 对象，不要任何其他文字：
{{
"公司名称": "提取的值",
"联系人": "提取的值",
...
}}
"""
        
        response_text, error = call_ai_api(prompt, image_data=image_data, custom_prompt=custom_prompt)
        if error:
            return None, error
        return parse_json_safely(response_text, "提取客户信息")

    def extract_document_content(doc):
        """提取 Word 文档内容"""
        content_parts = []
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                content_parts.append(f"[P{i}] {para.text.strip()}")
        
        for ti, table in enumerate(doc.tables):
            for ri, row in enumerate(table.rows):
                row_data = [c.text.strip() for c in row.cells]
                if any(row_data):
                    content_parts.append(f"[T{ti}-R{ri}] {' | '.join(row_data)}")
        
        return "\n".join(content_parts)

    def analyze_reference_document(doc, custom_prompt=None):
        """分析参考文档"""
        st.info("📄 正在提取文档内容...")
        doc_content = extract_document_content(doc)
        
        if len(doc_content) > 15000:
            doc_content = doc_content[:15000] + "\n...(内容过长，已截断)"
        
        st.info("🤖 AI 正在全面分析可变数据...")
        
        prompt = f"""
这是一份已填写好的文档内容。请提取所有"可变的、会随具体情况变化"的数据。

【文档内容】
{doc_content}

【任务】
提取所有会随不同情况变化的信息，如：公司名称、联系人、电话、地址、型号、数量、金额、日期等。

【输出格式】
只返回纯 JSON 对象：
{{
"公司名称": "具体值",
"联系人": "具体值",
...
}}
"""
        
        response_text, error = call_ai_api(prompt, custom_prompt=custom_prompt)
        if error:
            return None, error
        return parse_json_safely(response_text, "分析参考文档")

    def create_replacement_mapping(old_info, new_info, custom_prompt=None):
        """创建替换映射"""
        prompt = f"""
【参考数据（旧）】
{json.dumps(old_info, ensure_ascii=False, indent=2)}

【新数据】
{json.dumps(new_info, ensure_ascii=False, indent=2)}

【任务】
建立新旧数据的对应关系。如果新数据缺失则对应 null。
注意：键是旧文档中识别到的值，值是新数据中能匹配上的值。

【输出格式】
只返回纯 JSON 对象（值对值的映射）：
{{
"旧值1": "新值1",
"旧值2": "新值2",
"旧值3": null
}}
"""
        
        response_text, error = call_ai_api(prompt, custom_prompt=custom_prompt)
        if error:
            return None, error
        return parse_json_safely(response_text, "创建替换映射")

    def apply_replacements_to_document(doc, replacement_mapping):
        """应用替换到文档"""
        replace_count = 0
        replace_log = []
        
        # 核心：按长度降序排序，确保长字符串优先被替换
        sorted_map = sorted(replacement_mapping.items(), key=lambda x: len(str(x[0])), reverse=True)
        
        # 替换函数
        def replace_text_in_paragraph(paragraph, old_text, new_text):
            if old_text and new_text and old_text in paragraph.text:
                paragraph.text = paragraph.text.replace(old_text, new_text)
                return True
            return False

        for old_val, new_val in sorted_map:
            # 跳过空值替换
            if not old_val or new_val is None:
                continue
            
            # 强制转换为字符串
            old_val = str(old_val).strip()
            new_val = str(new_val).strip()

            if not old_val or not new_val:
                continue
            
            current_count = 0
            
            # 替换段落
            for p in doc.paragraphs:
                if replace_text_in_paragraph(p, old_val, new_val):
                    current_count += 1
            
            # 替换表格
            for t in doc.tables:
                for r in t.rows:
                    for c in r.cells:
                        for p in c.paragraphs:
                            if replace_text_in_paragraph(p, old_val, new_val):
                                current_count += 1
            
            # 替换页眉页脚
            for section in doc.sections:
                for header_footer in [section.header, section.footer]:
                    if header_footer:
                        for p in header_footer.paragraphs:
                            if replace_text_in_paragraph(p, old_val, new_val):
                                current_count += 1
            
            if current_count > 0:
                replace_count += current_count
                replace_log.append(f"✓ 替换 '{old_val}' → '{new_val}' ({current_count}处)")
        
        return replace_count, replace_log

    # ========== Streamlit 核心 UI 逻辑 ==========

    # 初始化 Session State
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'show_prompt_editor' not in st.session_state:
        st.session_state.show_prompt_editor = False

    for k in ['template_file', 'template_filename', 'old_customer_info', 'new_customer_info', 
              'replacement_mapping', 'uploaded_image_data', 'custom_replacements', 'current_prompt',
              'output_doc_bytes', 'replace_count', 'replace_log']: # 增加下载所需的状态
        if k not in st.session_state:
            st.session_state[k] = None if 'file' in k or 'image' in k or 'prompt' in k or 'bytes' in k else {}

    if st.session_state.custom_replacements is None:
        st.session_state.custom_replacements = []

    # 加载配置
    cfg = load_config()
    if 'api_type' not in st.session_state:
        st.session_state.api_type = cfg.get('api_type', 'gemini_custom')
    if 'api_key' not in st.session_state:
        st.session_state.api_key = cfg.get('api_key', '')
    if 'base_url' not in st.session_state:
        st.session_state.base_url = cfg.get('base_url', '')
    if 'model_name' not in st.session_state:
        st.session_state.model_name = cfg.get('model_name', '')
    if 'model_list' not in st.session_state:
        st.session_state.model_list = cfg.get('model_list', [])
    if 'prompt_settings' not in st.session_state:
        st.session_state.prompt_settings = cfg.get('prompt_settings', {})


    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.markdown("---")
        st.markdown("## ⚙️ API 配置")
        
        # 1. 选择 API 类型
        api_type_options = list(API_TYPES.keys())
        api_type_labels = [API_TYPES[k]["name"] for k in api_type_options]
        
        current_index = api_type_options.index(st.session_state.api_type) if st.session_state.api_type in api_type_options else 0
        
        selected_label = st.selectbox(
            "API 类型",
            options=api_type_labels,
            index=current_index,
            key="api_type_selector"
        )
        
        selected_type = api_type_options[api_type_labels.index(selected_label)]
        
        if selected_type != st.session_state.api_type:
            st.session_state.api_type = selected_type
            st.session_state.model_list = []
            st.session_state.model_name = ''
            save_config()
            st.rerun()
        
        # 2. API Key
        api_key_input = st.text_input(
            "API Key" + (" *必填" if "official" in st.session_state.api_type else " (可选)"),
            value=st.session_state.api_key,
            type="password",
            key="api_key_input"
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            save_config()
        
        # 3. Base URL（仅自定义需要）
        if API_TYPES[st.session_state.api_type]["needs_url"]:
            base_url_input = st.text_input(
                "Base URL *必填",
                value=st.session_state.base_url,
                placeholder="https://xxx.workers.dev",
                key="base_url_input"
            )
            
            if base_url_input != st.session_state.base_url:
                st.session_state.base_url = base_url_input
                save_config()
        
        st.markdown("---")
        
        # 4. 模型管理
        st.markdown("### 📋 模型管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 获取模型", use_container_width=True):
                with st.spinner("正在获取模型列表..."):
                    models, error = fetch_models_list(
                        st.session_state.api_type,
                        st.session_state.api_key,
                        st.session_state.base_url
                    )
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif models:
                        st.session_state.model_list = models
                        if not st.session_state.model_name or st.session_state.model_name not in models:
                            st.session_state.model_name = models[0] if models else ''
                        save_config()
                        st.success(f"✅ 获取成功！共 {len(models)} 个模型")
                        st.rerun()
                    else:
                        st.warning("未获取到模型列表")
        
        with col2:
            if st.button("🧪 测试连接", use_container_width=True):
                if not st.session_state.model_name:
                    st.warning("请先选择模型")
                else:
                    with st.spinner("正在测试..."):
                        success, message = test_api_connection(
                            st.session_state.api_type,
                            st.session_state.api_key,
                            st.session_state.base_url,
                            st.session_state.model_name
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        # 5. 模型选择
        if st.session_state.model_list:
            model_options = list(st.session_state.model_list)
            if st.session_state.model_name and st.session_state.model_name not in model_options:
                model_options.insert(0, st.session_state.model_name)
            
            current_model_index = model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
            
            selected_model = st.selectbox(
                "选择模型",
                options=model_options,
                index=current_model_index,
                key="model_selector"
            )
            
            if selected_model != st.session_state.model_name:
                st.session_state.model_name = selected_model
                save_config()
                st.rerun()
        else:
            # 手动输入模型名称
            model_input = st.text_input(
                "模型名称（手动输入）",
                value=st.session_state.model_name,
                placeholder="例如: gemini-1.5-flash",
                key="model_name_input"
            )
            
            if model_input != st.session_state.model_name:
                st.session_state.model_name = model_input
                save_config()
        
        st.markdown("---")
        
        # 6. 格式说明
        st.markdown("## 📄 格式说明")
        st.info("""
**仅支持 .docx 格式**
        """)

    # ==================== 主界面 ====================
    st.markdown('<div class="main-header">📄 智能文档填充工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">仿照模式 - AI学习已填好的文档</div>', unsafe_allow_html=True)

    # 顶部模型信息
    if st.session_state.api_key or st.session_state.base_url:
        api_name = API_TYPES.get(st.session_state.api_type, {}).get("name", "未知")
        st.markdown(f"""
        <div class="model-info">
            <span>✅ {api_name} | 模型: <code>{st.session_state.model_name or '未选择'}</code></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 请在左侧侧边栏配置 API")

    # 进度指示
    progress_cols = st.columns(5)
    steps = ["上传文档", "AI分析", "输入数据", "确认替换", "下载"]
    for i, col in enumerate(progress_cols, 1):
        with col:
            if st.session_state.step == i:
                st.markdown(f"### ✅ {steps[i-1]}")
            elif st.session_state.step > i:
                st.markdown(f"### ✓ {steps[i-1]}")
            else:
                st.markdown(f"### ⭕ {steps[i-1]}")

    st.markdown("---")

    # ==================== 步骤1: 上传参考文档 ====================
    if st.session_state.step >= 1:
        st.markdown("## 步骤1️⃣: 上传参考文档")
        st.info("💡 上传一份已经填写好的文档，AI会学习它的填写方式")
        
        uploaded_file = st.file_uploader(
            "选择已填好的文档（.docx）",
            type=['docx'],
            help="Word 文档格式",
            key="uploader_step1"
        )
        
        if uploaded_file and uploaded_file != st.session_state.template_file:
            st.session_state.template_file = uploaded_file
            st.session_state.template_filename = uploaded_file.name
            st.session_state.old_customer_info = {}
            st.session_state.replacement_mapping = {}
            st.session_state.output_doc_bytes = None
            st.success(f"✅ 已上传: {uploaded_file.name}")
            st.session_state.step = 1
            st.rerun()

        if st.session_state.template_file and st.session_state.step == 1:
            if st.button("下一步：AI分析 ➡️", type="primary"):
                st.session_state.step = 2
                st.rerun()

    # ==================== 步骤2: AI分析文档 ====================
    if st.session_state.step >= 2:
        st.markdown("## 步骤2️⃣: AI分析参考文档")
        
        if st.session_state.step == 2:
            with st.expander("💡 查看/编辑提示词（可选）", expanded=st.session_state.show_prompt_editor):
                st.markdown("### 临时自定义提示词")
                st.caption("仅在本次分析中生效")
                
                custom_prompt = st.text_area(
                    "自定义提示词",
                    value=st.session_state.current_prompt or "",
                    height=200,
                    placeholder="例如：\n• 重点识别技术参数和型号\n• 忽略通用条款\n• 优先提取数量和金额信息",
                    key="custom_prompt_analyze"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 应用此提示词", use_container_width=True, type="primary"):
                        st.session_state.current_prompt = custom_prompt.strip() if custom_prompt.strip() else None
                        st.success("✅ 已应用临时提示词")
                        st.session_state.show_prompt_editor = False
                        st.rerun()
                
                with col2:
                    if st.button("🔄 恢复默认", use_container_width=True):
                        st.session_state.current_prompt = None
                        st.success("✅ 已恢复默认配置")
                        st.rerun()
        
        if st.session_state.step == 2 and not st.session_state.old_customer_info and st.session_state.template_file:
            if not st.session_state.model_name or (st.session_state.api_key == '' and "official" in st.session_state.api_type):
                st.error("❌ 请先在侧边栏配置 API Key 和选择模型")
            else:
                with st.spinner("🚀 正在调用 AI 进行深度分析..."):
                    try:
                        st.session_state.template_file.seek(0) 
                        doc = Document(BytesIO(st.session_state.template_file.getvalue()))
                    except Exception as e:
                        st.error(f"❌ Word 文档加载失败，请检查文件是否损坏或格式是否为 .docx: {str(e)}")
                        st.session_state.step = 1
                        st.stop()
                    
                    old_info, error = analyze_reference_document(doc, st.session_state.current_prompt)
                    
                    if error:
                        st.error("❌ 分析失败")
                        with st.expander("🔍 查看详细错误信息", expanded=True):
                            st.markdown('<div class="error-detail">', unsafe_allow_html=True)
                            st.code(str(error), language='json')
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.session_state.old_customer_info = old_info
                        st.success("✅ 分析完成！")
                        st.rerun()
        
        if st.session_state.old_customer_info:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### 识别到的可变数据：")
            cols = st.columns(2)
            for idx, (field, value) in enumerate(st.session_state.old_customer_info.items()):
                with cols[idx % 2]:
                    st.markdown(f"**{field}:** `{value}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("⬅️ 重新上传"):
                    st.session_state.step = 1
                    st.session_state.old_customer_info = {}
                    st.session_state.current_prompt = None
                    st.session_state.template_file = None
                    st.rerun()
            with col2:
                if st.button("下一步：输入新数据 ➡️", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()

    # ==================== 步骤3: 输入新数据 ====================
    if st.session_state.step >= 3:
        st.markdown("## 步骤3️⃣: 输入新数据")
        st.info("💡 随意输入文本或上传图片，AI会自动识别格式并提取信息")
        
        with st.expander("💡 查看/编辑提示词（可选）"):
            st.markdown("### 临时自定义提示词")
            st.caption("仅在本次提取中生效")
            
            custom_prompt_extract = st.text_area(
                "自定义提示词",
                value="",
                height=150,
                placeholder="例如：\n• 重点识别联系方式\n• 提取所有电话和邮箱",
                key="custom_prompt_extract"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 文本输入")
            input_text = st.text_area(
                "新数据资料",
                height=300,
                placeholder="""例如：
上海普宙科技
张经理
mobile: 15912345678
浦东新区张江

或者：
Company: Shanghai Puzhou
Contact: Manager Zhang  
Tel 159-1234-5678"""
            )
        
        with col2:
            st.markdown("### 📷 图片识别")
            uploaded_image = st.file_uploader(
                "上传图片（名片/截图等）",
                type=['jpg', 'jpeg', 'png'],
                key="uploader_step3"
            )
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
                image_bytes = uploaded_image.getvalue()
                st.session_state.uploaded_image_data = base64.b64encode(image_bytes).decode()
            else:
                st.session_state.uploaded_image_data = None
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            if st.button("🤖 AI提取新数据", type="primary", use_container_width=True):
                if not input_text and not st.session_state.uploaded_image_data:
                    st.warning("请输入文本或上传图片")
                elif not st.session_state.model_name or (st.session_state.api_key == '' and "official" in st.session_state.api_type):
                    st.error("❌ 请先在侧边栏配置 API Key 和选择模型")
                else:
                    with st.spinner("AI正在提取新数据..."):
                        new_info, error = extract_customer_info_from_text(
                            input_text, 
                            st.session_state.uploaded_image_data, 
                            custom_prompt_extract
                        )
                        
                        if error:
                            st.error("❌ 新数据提取失败")
                            with st.expander("🔍 查看详细错误信息", expanded=True):
                                st.code(str(error), language='json')
                        elif new_info:
                            st.session_state.new_customer_info = new_info
                            st.session_state.step = 4
                            st.success("✅ 新数据提取完成！")
                            st.rerun()
                        else:
                            st.warning("AI未提取到任何有效信息，请修改输入后重试。")
        
        if st.session_state.new_customer_info:
            st.markdown('<div class="success-box" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown("### 已提取到的新数据：")
            cols = st.columns(2)
            for idx, (field, value) in enumerate(st.session_state.new_customer_info.items()):
                with cols[idx % 2]:
                    st.markdown(f"**{field}:** `{value}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.step > 3 and st.session_state.old_customer_info:
                if st.button("下一步：匹配替换映射 ➡️", type="primary", use_container_width=True):
                    st.session_state.step = 4
                    st.rerun()

    # ==================== 步骤4: 确认替换并生成新文档 ====================
    if st.session_state.step >= 4 and st.session_state.old_customer_info and st.session_state.new_customer_info:
        
        st.markdown("## 步骤4️⃣: 确认替换映射")
        
        if not st.session_state.replacement_mapping:
            with st.spinner("AI正在创建新旧数据映射..."):
                
                mapping, error = create_replacement_mapping(
                    st.session_state.old_customer_info, 
                    st.session_state.new_customer_info
                )
                
                if error:
                    st.error("❌ 映射创建失败")
                    with st.expander("🔍 查看详细错误信息", expanded=True):
                        st.code(str(error), language='json')
                else:
                    st.session_state.replacement_mapping = mapping
                    st.success("✅ 替换映射已生成！")
                    st.rerun()

        if st.session_state.replacement_mapping:
            st.markdown("### 📄 自动生成的新旧值替换映射")
            st.warning("⚠️ 请仔细核对，可在下方手动修改！")

            if 'editable_mapping' not in st.session_state:
                st.session_state.editable_mapping = [{"旧值 (Old)": old, "新值 (New)": new or ""} 
                                                     for old, new in st.session_state.replacement_mapping.items()]

            st.session_state.editable_mapping = st.data_editor(
                st.session_state.editable_mapping,
                column_config={
                    "旧值 (Old)": st.column_config.TextColumn("旧值 (Old)", disabled=True),
                    "新值 (New)": st.column_config.TextColumn("新值 (New)", help="空值将跳过替换")
                },
                num_rows="dynamic",
                use_container_width=True,
                key="replacement_editor"
            )
            
            if st.button("✅ 确认并生成文档", type="primary", use_container_width=True):
                
                final_mapping = {}
                for item in st.session_state.editable_mapping:
                    old_val = item.get("旧值 (Old)")
                    new_val = item.get("新值 (New)")
                    if old_val:
                        final_mapping[str(old_val)] = str(new_val) if new_val else None

                if not st.session_state.template_file:
                    st.error("❌ 模板文件缺失，请返回步骤 1 重新上传。")
                    st.stop()

                with st.spinner("💾 正在加载、替换并保存文档..."):
                    try:
                        st.session_state.template_file.seek(0)
                        template_stream = BytesIO(st.session_state.template_file.getvalue())
                        doc = Document(template_stream) 
                        
                        replace_count, replace_log = apply_replacements_to_document(
                            doc, final_mapping
                        )
                        
                        output_stream = BytesIO()
                        doc.save(output_stream)
                        output_stream.seek(0) 
                        
                        st.session_state.output_doc_bytes = output_stream.getvalue()
                        st.session_state.replace_count = replace_count
                        st.session_state.replace_log = replace_log
                        st.session_state.step = 5
                        st.success(f"✅ 文档生成完成！共替换 {replace_count} 处")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 文档生成或替换失败: {str(e)}")
                        st.session_state.output_doc_bytes = None
                        st.session_state.step = 4

    # ==================== 步骤5: 下载文档 ====================
    if st.session_state.step >= 5:
        st.markdown("## 步骤5️⃣: 下载新文档")
        
        if st.session_state.get('output_doc_bytes'):
            st.markdown(f"""
                <div class="success-box">
                    ### 🚀 文档已准备就绪
                    <p>✅ **文档生成完成！共替换 {st.session_state.replace_count} 处**</p>
                </div>
            """, unsafe_allow_html=True)
            
            original_name = st.session_state.template_filename
            base_name = os.path.splitext(original_name)[0]
            new_name = f"{base_name}_filled.docx"
            
            st.download_button(
                label="⬇️ 下载新文档 (.docx)",
                data=st.session_state.output_doc_bytes,
                file_name=new_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )

            with st.expander("🔍 查看替换详情"):
                if st.session_state.replace_log:
                    for log in st.session_state.replace_log:
                        st.text(log)
                else:
                    st.info("没有进行替换操作。")
            
            st.markdown("---")
            
            if st.button("重新开始新的任务", type="secondary"):
                st.session_state.step = 1
                for k in ['template_file', 'template_filename', 'old_customer_info', 'new_customer_info', 
                          'replacement_mapping', 'uploaded_image_data', 'current_prompt', 'output_doc_bytes', 
                          'replace_count', 'replace_log', 'editable_mapping']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

# ==============================================================================
#                      主程序入口 (包含密码验证)
# ==============================================================================

# 设置页面配置 (仅在应用逻辑外部设置一次)
st.set_page_config(
    page_title="智能文档填充工具 - 登录",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 配置认证器 ---
# 默认用户：document_user，密码：password
# 密码 'password' 的哈希值（已使用 bcrypt 生成）
hashed_passwords = ['\$2b\$12\$R.32u.L.V/iH4H62hX9y4.2c6dF6j/g7e8JpWzY5Xq3hY0hP5J3xG']

config = {
    'cookie': {
        'name': 'document_filler_cookie',
        'key': 'random_long_signature_key_for_security_1234567890', # 实际部署时请换成随机长字符串
        'expiry_days': 30
    },
    'credentials': {
        'usernames': {
            'document_user': { 
                'email': 'user@example.com',
                'name': '文档填充用户',
                'password': hashed_passwords[0]
            }
        }
    }
}

# --- 2. 初始化认证器 ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- 3. 登录逻辑 ---
# 侧边栏的登录/登出需要在主程序中进行
st.sidebar.title("🔐 登录/登出")
name, authentication_status, username = authenticator.login('Login', 'main')


if st.session_state["authentication_status"]:
    # 登录成功
    st.sidebar.success(f'欢迎回来, {st.session_state["name"]}!')
    authenticator.logout('退出登录', 'sidebar')

    # 运行应用的主体功能
    run_app()
    
elif st.session_state["authentication_status"] is False:
    # 登录失败
    st.error('❌ 用户名或密码错误')
    st.sidebar.markdown("---")
    st.sidebar.info("💡 默认用户名：`document_user`，密码：`password`")
    
elif st.session_state["authentication_status"] is None:
    # 尚未登录
    st.warning('⚠️ 请先在侧边栏输入您的用户名和密码以继续')
    st.markdown("---")
    st.info("💡 默认用户名：`document_user`，密码：`password`")
    st.markdown("### 📄 智能文档填充工具")
    st.markdown("登录后即可使用全部功能。")