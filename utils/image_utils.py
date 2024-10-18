import imageio

# 图像转换成视频
def convert_images_to_video(input_path, output_path, videofilename, size=800):
    '''
        param input_path: 输入路径
        param output_path: 输出路径
        size，图片的数量 默认为800张
        return: 无返回值
    '''
    # 创建一个空列表，用于存储所有图片的文件名
    image_files = []
    # 遍历数组的第一个维度，生成图片文件名并添加到列表中
    for i in range(1, size + 1):
        image_file = input_path + f"//de_{i}.jpg"
        image_files.append(image_file)
    # 使用imageio库将所有图片合成为一个视频文件
    with imageio.get_writer(output_path + videofilename, mode='I') as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

