# GPT-SoVITS-For-Intel

这是一个基于原版 GPT-SoVITS 项目修改的整合包，实际是**强行在原版 Nvidia 整合包上安装支持 Intel 显卡的 PyTorch 版本**。

* 使用的 PyTorch 版本为 **2.7.10**。
* 经过几个小时的代码大改，sovits训练只将cuda改成xpu就成了，gpt训练由于pytorch_lighting不支持xpu，所以强行改成了直接使用pytorch，目前训练问题也已解决，a系列和b系列显卡都可以训练和推理了。
* **请注意**：因为只改了v4训练，**使用该整合包模型记得都选v4的**，毕竟有v4了应该也没人会用v1和v2，并且由于分布式训练总是报错，所以删掉了分布式训练代码，**整合包仅支持1张intel显卡训练**，另外uvr5也没有去做适配，但是去uvr5官网下载他们的软件，可以直接支持intel gpu，而且模型更多，**所以都去下载uvr5软件进行人声分离和去噪**。

**下载地址：**

* [网页直链下载（6 MB/s）](https://www.modelscope.cn/models/Sakuya999/GPT-SoVITS-For-Intel/resolve/master/GPT-SoVITS-For-Intel.zip)
* [百度网盘](https://pan.baidu.com/s/15dQEE25pwTn5tKm7TsGFGA?pwd=1616)
* [夸克网盘](https://pan.quark.cn/s/352c55104048#/list/share)
* [123网盘](https://www.123865.com/s/0p0Mjv-oXggh)

**参考项目：**

* GPT-SoVITS For Intel参考：https://github.com/liyisker/GPT-SoVITS-Intel-Arc-GPU
* 源项目：https://github.com/RVC-Boss/GPT-SoVITS
* Intel PyTorch：https://pytorch-extension.intel.com/
