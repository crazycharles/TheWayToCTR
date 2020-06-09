Hello!  
这份文档记录了我自己对FFM算法原理的理解，以及基于tensorflow的代码实现。  
ffm.py就是我的ffm算法实现，你可以自己扩展功能。

preprocess.py是用来处理原始的数据的，作用在于将原始数据转换为one-hot类型的。

## :sunglasses:文件说明
文件名|内容说明
-|-
ffm.py|ffm算法实现，可以自行扩展功能
main.py|流程主函数
preprocess.py|对原始数据进行one-hot编码
train_200.csv|原始的200个数据
train_200_converted.csv|one-hot处理之后的数据
