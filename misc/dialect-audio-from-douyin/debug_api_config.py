#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
讯飞API配置调试工具
帮助诊断API配置问题
"""

import re
import json


def validate_api_config(app_id, api_key, api_secret):
    """验证API配置格式"""
    print("验证API配置格式...")
    
    # 验证APPID
    if not app_id:
        print("❌ APPID不能为空")
        return False
    
    if len(app_id) != 8:
        print(f"❌ APPID长度应为8位，当前为{len(app_id)}位")
        return False
    
    if not app_id.isalnum():
        print("❌ APPID应只包含数字和字母")
        return False
    
    print("✓ APPID格式正确")
    
    # 验证APIKey
    if not api_key:
        print("❌ APIKey不能为空")
        return False
    
    if len(api_key) != 32:
        print(f"❌ APIKey长度应为32位，当前为{len(api_key)}位")
        return False
    
    if not re.match(r'^[a-f0-9]{32}$', api_key):
        print("❌ APIKey格式不正确，应为32位十六进制字符")
        return False
    
    print("✓ APIKey格式正确")
    
    # 验证APISecret
    if not api_secret:
        print("❌ APISecret不能为空")
        return False
    
    if len(api_secret) != 32:
        print(f"❌ APISecret长度应为32位，当前为{len(api_secret)}位")
        return False
    
    if not re.match(r'^[a-f0-9]{32}$', api_secret):
        print("❌ APISecret格式不正确，应为32位十六进制字符")
        return False
    
    print("✓ APISecret格式正确")
    
    return True


def check_common_issues():
    """检查常见问题"""
    print("\n常见问题检查:")
    print("1. 确认您的讯飞账户状态:")
    print("   - 登录 https://console.xfyun.cn/ 检查账户状态")
    print("   - 确认账户有足够的余额或免费额度")
    print("   - 检查应用状态是否正常")
    
    print("\n2. 确认服务配置:")
    print("   - 确认已在应用中添加了'语音听写（流式版）'服务")
    print("   - 检查服务状态是否为'已开通'")
    print("   - 确认IP白名单设置（如有）")
    
    print("\n3. 网络连接:")
    print("   - 确认能够访问 iat.cn-huabei-1.xf-yun.com")
    print("   - 检查防火墙设置")
    print("   - 确认没有代理阻止WebSocket连接")


def generate_test_code(app_id, api_key, api_secret):
    """生成测试代码"""
    code = f'''
# 在 spark_iat_recognition.py 中替换这些值:
app_id = "{app_id}"
api_key = "{api_key}"
api_secret = "{api_secret}"
'''
    return code


def main():
    """主函数"""
    print("讯飞API配置调试工具")
    print("=" * 50)
    
    # 获取用户输入
    app_id = input("请输入APPID: ").strip()
    api_key = input("请输入APIKey: ").strip()
    api_secret = input("请输入APISecret: ").strip()
    
    print("\n" + "=" * 50)
    
    # 验证配置格式
    if validate_api_config(app_id, api_key, api_secret):
        print("\n✓ 配置格式验证通过")
        
        # 生成测试代码
        print("\n配置代码:")
        print(generate_test_code(app_id, api_key, api_secret))
        
        print("接下来请运行 test_spark_api.py 进行连接测试")
    else:
        print("\n❌ 配置格式验证失败")
        print("请检查您的API配置信息")
    
    # 显示常见问题
    check_common_issues()
    
    print("\n" + "=" * 50)
    print("如果问题仍然存在，请:")
    print("1. 重新生成API密钥")
    print("2. 检查讯飞开放平台的服务状态")
    print("3. 联系讯飞技术支持")


if __name__ == "__main__":
    main() 