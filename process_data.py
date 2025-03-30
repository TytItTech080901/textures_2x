import os

# 定义函数：获取文件夹中所有PNG图片的相对路径
def get_png_files(folder):
    png_files = set()
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):  # 检查文件是否为PNG格式（忽略大小写）
                rel_path = os.path.relpath(os.path.join(root, file), folder)
                png_files.add(rel_path)
    return png_files

# 请替换为实际的文件夹路径
input_folder = 'dataset/input'  # 替换为您的input文件夹路径
output_folder = 'dataset/target'  # 替换为您的output文件夹路径

# 获取input和output文件夹中所有PNG图片的相对路径
input_png_files = get_png_files(input_folder)
output_png_files = get_png_files(output_folder)

# 找出input中没有对应output的图片
unpaired_files = input_png_files - output_png_files

# 删除input文件夹中不配对的图片
for rel_path in unpaired_files:
    file_path = os.path.join(input_folder, rel_path)
    os.remove(file_path)
    print(f"已删除: {file_path}")

# 统计input文件夹中剩余的PNG图片数量
remaining_input_png_files = get_png_files(input_folder)
print(f"最终input文件夹中的PNG图片数量: {len(remaining_input_png_files)}")
