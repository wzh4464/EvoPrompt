#!/usr/bin/env python3
"""
SVEN-style LLM Client for EvoPrompt
基于SVEN的API调用方式，支持多种API base和环境配置
"""

import json
import os
import requests
import time
from typing import List, Dict, Optional, Union
from pathlib import Path


def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load environment variables
load_env_vars()

# API Configuration from environment
API_BASE = os.getenv("API_BASE_URL", "https://newapi.pockgo.com/v1")
API_KEY = os.getenv("API_KEY", "")
BACKUP_API_BASE = os.getenv("BACKUP_API_BASE_URL", "https://newapi.aicohere.org/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")


class SVENLLMClient:
    """SVEN风格的LLM客户端"""
    
    def __init__(self, api_base: str = None, api_key: str = None, model_name: str = None):
        self.api_base = api_base or API_BASE
        self.api_key = api_key or API_KEY
        self.model_name = model_name or MODEL_NAME
        self.backup_api_base = BACKUP_API_BASE
        
        if not self.api_key:
            raise ValueError("API_KEY not found. Please set it in .env file or environment variable.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """发送API请求"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 尝试主API，失败则尝试备用API
        apis_to_try = [self.api_base, self.backup_api_base]
        
        for api_base in apis_to_try:
            try:
                response = self.session.post(
                    f"{api_base}/chat/completions",
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
                
            except Exception as e:
                print(f"API call failed for {api_base}: {e}")
                if api_base == apis_to_try[-1]:  # 最后一个API也失败了
                    raise Exception(f"All API endpoints failed. Last error: {e}")
                continue  # 尝试下一个API
        
        raise Exception("No API endpoints available")
    
    def query_single(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """单次查询"""
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages, temperature, max_tokens)
    
    def query_batch(self, prompts: List[str], temperature: float = 0.1, max_tokens: int = 1000, 
                   delay: float = 0.1) -> List[str]:
        """批量查询"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.query_single(prompt, temperature, max_tokens)
                results.append(result)
                
                # 进度显示
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(prompts)} queries")
                
                # 速率限制
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Query {i+1} failed: {e}")
                results.append("error")
        
        return results
    
    def paraphrase(self, sentence: Union[str, List[str]], temperature: float = 0.7) -> Union[str, List[str]]:
        """重新表述句子"""
        if isinstance(sentence, list):
            prompts = [
                f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput: {s}\nOutput:"
                for s in sentence
            ]
            return self.query_batch(prompts, temperature=temperature)
        else:
            prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput: {sentence}\nOutput:"
            return self.query_single(prompt, temperature=temperature)


def sven_llm_init(api_base: str = None, api_key: str = None, model_name: str = None) -> SVENLLMClient:
    """初始化SVEN风格的LLM客户端"""
    return SVENLLMClient(api_base, api_key, model_name)


def sven_llm_query(data: Union[str, List[str]], client: SVENLLMClient, task: bool = False, 
                  temperature: float = 0.1, **kwargs) -> Union[str, List[str]]:
    """
    SVEN风格的LLM查询函数
    
    Args:
        data: 单个提示或提示列表
        client: SVEN LLM客户端
        task: 是否为任务导向（会截断多段回答）
        temperature: 温度参数
        **kwargs: 其他参数
    
    Returns:
        单个回答或回答列表
    """
    if isinstance(data, list):
        results = client.query_batch(data, temperature=temperature, **kwargs)
        if task:
            # 任务导向，只取第一段
            results = [str(r).strip().split("\n\n")[0] for r in results]
        return results
    else:
        result = client.query_single(data, temperature=temperature, **kwargs)
        if task:
            result = result.split("\n\n")[0]
        return result


# 兼容性函数，保持与原EvoPrompt接口一致
def llm_init(**kwargs) -> SVENLLMClient:
    """兼容性函数"""
    return sven_llm_init()


def llm_query(data, client, type=None, task=False, **config):
    """兼容性函数"""
    return sven_llm_query(data, client, task=task, **config)


if __name__ == "__main__":
    # 测试代码
    print("Testing SVEN LLM Client...")
    
    try:
        client = sven_llm_init()
        
        # 单次查询测试
        test_prompt = "Explain what a buffer overflow vulnerability is in one sentence."
        result = sven_llm_query(test_prompt, client)
        print(f"Single query result: {result}")
        
        # 批量查询测试
        test_prompts = [
            "What is SQL injection?",
            "What is XSS?",
            "What is CSRF?"
        ]
        results = sven_llm_query(test_prompts, client)
        print(f"Batch query results: {results}")
        
        print("SVEN LLM Client test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")