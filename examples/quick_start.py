#!/usr/bin/env python3
"""
Diffusion模型快速入门示例
=========================

这个脚本演示了如何使用diffusers库快速生成图像。
适合初学者快速体验扩散模型的基本功能。

作者: Diffusion教程团队
日期: 2024年
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
import warnings
warnings.filterwarnings('ignore')

def setup_device():
    """
    配置计算设备
    
    返回:
        device: 可用的计算设备 (cuda/mps/cpu)
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():  # Apple Silicon
        device = "mps"
        print("✅ 使用Apple Silicon GPU")
    else:
        device = "cpu"
        print("⚠️  使用CPU (速度较慢)")
    
    return device

def load_model(device):
    """
    加载预训练的Stable Diffusion模型
    
    参数:
        device: 计算设备
    
    返回:
        pipeline: 配置好的扩散模型管道
    """
    print("🔄 正在加载Stable Diffusion模型...")
    
    # 使用Hugging Face上的预训练模型
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        # 创建管道
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # 为了演示简单起见，禁用安全检查
            requires_safety_checker=False
        )
        
        # 使用更快的调度器
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        # 移动到指定设备
        pipeline = pipeline.to(device)
        
        # 如果使用GPU，启用内存优化
        if device == "cuda":
            pipeline.enable_attention_slicing()  # 节省显存
            pipeline.enable_xformers_memory_efficient_attention()  # 加速注意力计算
        
        print("✅ 模型加载完成!")
        return pipeline
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 尝试使用CPU或检查网络连接")
        return None

def generate_image(pipeline, prompt, negative_prompt="", steps=20, guidance_scale=7.5):
    """
    生成图像
    
    参数:
        pipeline: 扩散模型管道
        prompt: 正向提示词 (描述想要生成的内容)
        negative_prompt: 负向提示词 (描述不想要的内容)
        steps: 推理步数 (更多步数 = 更高质量, 但更慢)
        guidance_scale: 引导强度 (更高 = 更贴近提示词)
    
    返回:
        image: 生成的PIL图像
    """
    print(f"🎨 正在生成图像...")
    print(f"   提示词: {prompt}")
    print(f"   推理步数: {steps}")
    print(f"   引导强度: {guidance_scale}")
    
    # 设置随机种子以便结果可复现
    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    
    try:
        # 生成图像
        with torch.autocast(pipeline.device.type):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,
                width=512
            )
        
        image = result.images[0]
        print("✅ 图像生成完成!")
        return image
        
    except Exception as e:
        print(f"❌ 图像生成失败: {e}")
        return None

def save_and_display_image(image, prompt, output_dir="outputs"):
    """
    保存并显示生成的图像
    
    参数:
        image: PIL图像
        prompt: 用于生成的提示词
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt[:50]  # 限制长度
    filename = f"{safe_prompt}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图像
    image.save(filepath)
    print(f"💾 图像已保存到: {filepath}")
    
    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"生成的图像\n提示词: {prompt}", fontsize=12, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def demo_basic_generation():
    """
    基础图像生成演示
    """
    print("=" * 60)
    print("🚀 Diffusion模型快速入门演示")
    print("=" * 60)
    
    # 1. 设置设备
    device = setup_device()
    
    # 2. 加载模型
    pipeline = load_model(device)
    if pipeline is None:
        return
    
    # 3. 定义示例提示词
    demo_prompts = [
        {
            "prompt": "a beautiful landscape with mountains and lake, sunset, highly detailed",
            "negative_prompt": "ugly, blurry, low quality",
            "description": "美丽的山湖风景"
        },
        {
            "prompt": "a cute cat wearing a wizard hat, magical, fantasy art",
            "negative_prompt": "scary, dark, low quality",
            "description": "戴巫师帽的可爱猫咪"
        },
        {
            "prompt": "a futuristic city with flying cars, cyberpunk style, neon lights",
            "negative_prompt": "old, ancient, low quality",
            "description": "赛博朋克未来城市"
        }
    ]
    
    # 4. 生成图像
    print(f"\n📝 将生成 {len(demo_prompts)} 张示例图像:")
    for i, prompt_config in enumerate(demo_prompts, 1):
        print(f"\n--- 示例 {i}/{len(demo_prompts)}: {prompt_config['description']} ---")
        
        image = generate_image(
            pipeline=pipeline,
            prompt=prompt_config["prompt"],
            negative_prompt=prompt_config["negative_prompt"],
            steps=20,
            guidance_scale=7.5
        )
        
        if image:
            save_and_display_image(
                image, 
                f"{i}_{prompt_config['description']}"
            )

def demo_parameter_comparison():
    """
    参数对比演示 - 展示不同参数对生成结果的影响
    """
    print("\n" + "=" * 60)
    print("🔬 参数对比演示")
    print("=" * 60)
    
    device = setup_device()
    pipeline = load_model(device)
    if pipeline is None:
        return
    
    base_prompt = "a serene garden with blooming flowers"
    
    # 对比不同的引导强度
    guidance_scales = [5.0, 7.5, 15.0]
    
    print(f"\n📊 对比不同引导强度的效果:")
    print(f"提示词: {base_prompt}")
    
    for guidance in guidance_scales:
        print(f"\n🎛️  引导强度: {guidance}")
        
        image = generate_image(
            pipeline=pipeline,
            prompt=base_prompt,
            steps=20,
            guidance_scale=guidance
        )
        
        if image:
            save_and_display_image(
                image, 
                f"guidance_{guidance}_{base_prompt[:20]}"
            )

def interactive_generation():
    """
    交互式图像生成
    """
    print("\n" + "=" * 60)
    print("🎮 交互式图像生成")
    print("=" * 60)
    
    device = setup_device()
    pipeline = load_model(device)
    if pipeline is None:
        return
    
    print("\n💡 现在您可以输入自己的提示词来生成图像!")
    print("提示: 使用英文描述，越详细越好")
    print("输入 'quit' 退出\n")
    
    while True:
        try:
            # 获取用户输入
            user_prompt = input("🎨 请输入提示词: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用!")
                break
            
            if not user_prompt:
                print("❌ 提示词不能为空，请重新输入")
                continue
            
            # 可选的负向提示词
            negative_prompt = input("🚫 负向提示词 (可选，直接回车跳过): ").strip()
            
            # 生成图像
            image = generate_image(
                pipeline=pipeline,
                prompt=user_prompt,
                negative_prompt=negative_prompt,
                steps=20,
                guidance_scale=7.5
            )
            
            if image:
                save_and_display_image(image, user_prompt)
                
                # 询问是否继续
                continue_choice = input("\n🔄 是否继续生成? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '是']:
                    print("👋 感谢使用!")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 感谢使用!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            continue

def main():
    """
    主函数 - 展示不同的演示模式
    """
    print("🎯 Diffusion模型快速入门")
    print("选择演示模式:")
    print("1. 基础图像生成演示")
    print("2. 参数对比演示") 
    print("3. 交互式生成")
    print("4. 全部运行")
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == "1":
            demo_basic_generation()
        elif choice == "2":
            demo_parameter_comparison()
        elif choice == "3":
            interactive_generation()
        elif choice == "4":
            demo_basic_generation()
            demo_parameter_comparison()
            interactive_generation()
        else:
            print("❌ 无效选择，运行基础演示")
            demo_basic_generation()
            
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
    
    print("\n🎉 演示完成! 更多教程请查看项目文档。")

if __name__ == "__main__":
    main()