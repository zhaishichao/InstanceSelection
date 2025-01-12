import imageio.v2 as imageio
import matplotlib.pyplot as plt
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

def histogram(x,data,title,save_path):
    # 设置可显示中文宋体
    plt.rcParams['font.family'] = 'STZhongsong'
    plt.title(title)
    # plt.grid(ls="--", alpha=0.5)
    plt.bar(x, data, width=0.4)
    for a, b in zip(x, data):
        plt.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=12)
    plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close()


# 绘制Pareto Front曲线
def plot_front(fronts, gen, title):
    """绘制当前代非支配排序的第一等级前沿"""
    fitnesses = [ind.fitness.values for ind in fronts]
    plt.scatter(*zip(*fitnesses), marker='o', label=f"Generation {gen}")
    plt.title(title)
    plt.xlabel("G-mean")
    plt.ylabel("mAUC")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()