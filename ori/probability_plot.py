import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))


def two_classify(type1, type2, probability1, probability2):
    x = [type1, type2]
    y = [probability1, probability2]
    # auto_label(plt.bar(range(len(y)), y, color='rgb', tick_label=x))
    plt.bar(range(len(y)), y, color='rgb', tick_label=x)
    plt.savefig("/home/fan/Desktop/pro.png")
    plt.show()


if __name__ == '__main__':
    two_classify("AD", "NC", 0.7, 0.3)

# # 显示高度
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))
#
#
# name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# num_list = [33, 44, 53, 16, 11, 17, 17, 10]
# autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
# plt.show()