#!/usr/bin/env python3
"""
模型客户端封装，支持官方API和兼容中转站
支持自定义API Base URL，适配各种OpenAI兼容中转服务
"""
import os
import json
import time
from typing import Dict, Any, Optional
import openai
import anthropic
import dashscope

class BaseModelClient:
    """模型客户端基类"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_id = model_config['model_id']
        self.model_name = model_config['model_name']
        self.provider = model_config['provider']
        self.temperature = model_config.get('temperature', 0.0)
        self.max_output_tokens = model_config.get('max_output_tokens', 4096)
        self.api_endpoint = model_config.get('api_endpoint', '')
        self.timeout = model_config.get('timeout', 120)
        self.max_retries = model_config.get('max_retries', 3)
    
    def _resolve_api_base(self, env_key: str, default_url: str) -> str:
        """解析API Base URL：优先使用环境变量，否则使用默认值"""
        custom_base = os.getenv(env_key, '').strip()
        if custom_base:
            return custom_base
        return default_url
    
    def _resolve_api_key(self, env_key: str) -> str:
        """解析API Key"""
        api_key = os.getenv(env_key, '').strip()
        if not api_key:
            raise ValueError(f"环境变量 {env_key} 未设置，请在 .env 文件中配置")
        return api_key
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """生成回答，返回标准格式"""
        raise NotImplementedError

class OpenAIClient(BaseModelClient):
    """OpenAI API客户端（支持自定义API Base和中转站）"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = self._resolve_api_key(model_config['api_key_env'])
        api_base = self._resolve_api_base(
            model_config.get('api_base_env', ''),
            model_config.get('api_endpoint', 'https://api.openai.com/v1')
        )
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout
        )
        self.actual_api_base = api_base
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用OpenAI API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=1.0
            )
            
            end_time = time.time()
            
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return {
                "success": True,
                "raw_output": response.choices[0].message.content,
                "usage": usage,
                "latency": end_time - start_time,
                "model_version": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "api_base": self.actual_api_base
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": None,
                "api_base": self.actual_api_base
            }

class AnthropicClient(BaseModelClient):
    """Anthropic API客户端（使用 httpx 直接请求，避免 SDK 自动拼接路径）"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = self._resolve_api_key(model_config['api_key_env'])
        api_base = self._resolve_api_base(
            model_config.get('api_base_env', ''),
            model_config.get('api_endpoint', 'https://api.anthropic.com')
        )
        
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')  # 移除末尾斜杠
        self.actual_api_base = api_base
        self.max_retries = model_config.get('max_retries', 5)  # 增加重试次数
        self.retry_delay = model_config.get('retry_delay', 2)  # 重试延迟
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用Anthropic API（使用 httpx 直接请求，带重试机制）"""
        import httpx
        
        start_time = time.time()
        
        # 构造 Anthropic Messages API 请求
        url = f"{self.api_base}/v1/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "stream": True  # 启用流式输出
        }
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                full_output = []
                
                # 使用 httpx 流式请求，增加超时时间
                with httpx.Client(timeout=300.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code == 200:
                            # 处理 SSE 流
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    data = line[6:]  # 移除 "data: " 前缀
                                    if data == "[DONE]":
                                        break
                                    
                                    try:
                                        event = json.loads(data)
                                        event_type = event.get("type", "")
                                        
                                        if event_type == "content_block_delta":
                                            delta = event.get("delta", {})
                                            text = delta.get("text", "")
                                            if text:
                                                full_output.append(text)
                                                print(text, end="", flush=True)  # 流式输出到终端
                                        
                                        elif event_type == "message_stop":
                                            break
                                    except json.JSONDecodeError:
                                        continue
                            
                            print()  # 输出换行
                            
                            end_time = time.time()
                            raw_output = "".join(full_output).strip()

                            if not raw_output:
                                error_msg = "HTTP 200，但流式响应未包含文本输出"
                                if attempt < self.max_retries - 1:
                                    wait_time = self.retry_delay * (2 ** attempt)
                                    print(f"\n⚠️ 空输出重试 {attempt+1}/{self.max_retries}，等待 {wait_time}秒...")
                                    time.sleep(wait_time)
                                    continue
                                return {
                                    "success": False,
                                    "error": f"{error_msg}（重试{self.max_retries}次后失败）",
                                    "raw_output": None,
                                    "latency": end_time - start_time,
                                    "api_base": self.actual_api_base
                                }

                            return {
                                "success": True,
                                "raw_output": raw_output,
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0
                                },
                                "latency": end_time - start_time,
                                "model_version": self.model_name,
                                "finish_reason": "stop",
                                "api_base": self.actual_api_base
                            }
                        
                        elif response.status_code in [503, 502, 429]:
                            # 服务不可用或速率限制，重试
                            error_body = response.read().decode('utf-8')
                            error_msg = f"HTTP {response.status_code}: {error_body[:200]}"
                            
                            if attempt < self.max_retries - 1:
                                # 指数退避：2, 4, 8, 16, 32秒
                                wait_time = self.retry_delay * (2 ** attempt)
                                print(f"\n⚠️ 重试 {attempt+1}/{self.max_retries}，等待 {wait_time}秒... 错误: {error_msg[:100]}")
                                time.sleep(wait_time)
                                continue
                            else:
                                return {
                                    "success": False,
                                    "error": f"{error_msg} (重试{self.max_retries}次后失败)",
                                    "raw_output": None,
                                    "api_base": self.actual_api_base
                                }
                        else:
                            # 其他错误，不重试
                            error_body = response.read().decode('utf-8')
                            return {
                                "success": False,
                                "error": f"HTTP {response.status_code}: {error_body[:200]}",
                                "raw_output": None,
                                "api_base": self.actual_api_base
                            }
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"\n⚠️ 异常重试 {attempt+1}/{self.max_retries}，等待 {wait_time}秒... 错误: {str(e)[:100]}")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"{str(e)} (重试{self.max_retries}次后失败)",
                        "raw_output": None,
                        "api_base": self.actual_api_base
                    }
        
        # 理论上不会到达这里
        return {
            "success": False,
            "error": "未知错误",
            "raw_output": None,
            "api_base": self.actual_api_base
        }

class OpenAICompatibleClient(BaseModelClient):
    """通用OpenAI兼容客户端（用于DeepSeek、中转站等）
    
    只要API兼容OpenAI格式，都可以通过这个客户端调用。
    支持的中转站格式：
    - OpenAI兼容中转站
    - DeepSeek官方API
    - 其他兼容OpenAI格式的第三方服务
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = self._resolve_api_key(model_config['api_key_env'])
        api_base = self._resolve_api_base(
            model_config.get('api_base_env', ''),
            model_config.get('api_endpoint', '')
        )
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout
        )
        self.actual_api_base = api_base
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用兼容API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=1.0
            )
            
            end_time = time.time()
            
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                if response.usage and not isinstance(response.usage, str):
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
            except (AttributeError, TypeError):
                pass
            
            return {
                "success": True,
                "raw_output": response.choices[0].message.content,
                "usage": usage,
                "latency": end_time - start_time,
                "model_version": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "api_base": self.actual_api_base
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": None,
                "api_base": self.actual_api_base
            }

class DashScopeClient(BaseModelClient):
    """阿里云DashScope API客户端（支持自定义API Base）"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = self._resolve_api_key(model_config['api_key_env'])
        
        # DashScope SDK使用api_key属性
        dashscope.api_key = api_key
        
        # 支持自定义base_url
        api_base = os.getenv(model_config.get('api_base_env', ''), '').strip()
        if api_base:
            dashscope.base_api_url = api_base
        
        self.actual_api_base = api_base or "https://dashscope.aliyuncs.com/api/v1"
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用DashScope API"""
        start_time = time.time()
        
        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=1.0,
                result_format='message'
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "raw_output": response.output.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    "latency": end_time - start_time,
                    "model_version": response.model,
                    "finish_reason": response.output.choices[0].finish_reason,
                    "api_base": self.actual_api_base
                }
            else:
                return {
                    "success": False,
                    "error": f"API错误: {response.code} - {response.message}",
                    "raw_output": None,
                    "api_base": self.actual_api_base
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": None,
                "api_base": self.actual_api_base
            }

class LMStudioClient(BaseModelClient):
    """LM Studio本地模型客户端（OpenAI兼容格式）"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        # LM Studio本地API
        api_key = os.getenv(model_config.get('api_key_env', ''), 'lm-studio')
        api_base = self._resolve_api_base(
            model_config.get('api_base_env', ''),
            model_config.get('api_base', 'http://localhost:1234/v1')
        )
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.timeout
        )
        self.actual_api_base = api_base
    
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用LM Studio本地API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=1.0
            )
            
            end_time = time.time()
            
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return {
                "success": True,
                "raw_output": response.choices[0].message.content,
                "usage": usage,
                "latency": end_time - start_time,
                "model_version": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "api_base": self.actual_api_base
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": None,
                "api_base": self.actual_api_base
            }

def get_client(model_config: Dict[str, Any]) -> BaseModelClient:
    """根据配置获取对应的客户端
    
    支持的provider类型：
    - 'openai': OpenAI官方API 或 OpenAI兼容中转站
    - 'anthropic': Anthropic官方API 或 Anthropic兼容中转站
    - 'deepseek': DeepSeek API（OpenAI兼容格式）
    - 'alibaba': 阿里云DashScope API
    - 'openai_compatible': 任何OpenAI兼容的第三方API
    - 'lmstudio': LM Studio本地模型
    """
    provider = model_config['provider'].lower()
    
    if provider == 'openai':
        return OpenAIClient(model_config)
    elif provider == 'anthropic':
        return AnthropicClient(model_config)
    elif provider == 'deepseek':
        return OpenAICompatibleClient(model_config)
    elif provider == 'alibaba':
        return DashScopeClient(model_config)
    elif provider == 'openai_compatible':
        return OpenAICompatibleClient(model_config)
    elif provider == 'lmstudio':
        return LMStudioClient(model_config)
    else:
        raise ValueError(f"不支持的模型提供方: {provider}")