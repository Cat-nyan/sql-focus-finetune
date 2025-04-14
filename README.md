# 🚀 **XiYanSQL-QwenCoder-7B-Finetune**

> **描述：**  
本代码仓库提供 XiYanSQL-QwenCoder-7B-Finetune 模型的完整微调及应用流程，使用了LoRA微调方式，使模型更专注text2sql，微调过程结合 SwanLab 可视化工具，全程记录并监控训练过程中的关键指标，如损失、学习率变化等，帮助用户更直观的感受模型性能变化。在模型的应用方面，大模型可直接访问数据库，并通过M-Schema工具进一步简化prompt的编写，降低使用门槛。
 
## 一、环境准备
项目基于大语言模型通过LoRA微调方式来提升Text-to-SQL的效果。   

### 1.1 数据集
本项目使用了**Spider**数据集:
- [Spider](https://yale-lily.github.io/spider): 一个跨域的复杂text2sql数据集，包含了10,181条自然语言问句、分布在200个独立数据库中的5,693条SQL，内容覆盖了138个不同的领域。[下载链接](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ)（本项目已提供处理后的数据集，可直接使用）

### 1.2、基座模型
模型基于开源析言SQL-通义千问系列模型[析言SQL-通义千问-7B](https://modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2502):该模型在SQL生成方面表现出色，可以直接用于文本到SQL任务，或作为微调SQL模型的更好起点。

💻[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-7B-2502) 🤗[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2502)

**模型将会在主程序运行时自动拉取，无需再手动下载。**

### 1.3、系统要求
- **操作系统**: Linux | Windows(Linux子系统WSL2) | MacOS
- **Python版本**: 3.10+
- **CUDA**: 建议使用 CUDA 12.1 及以上

### 1.4 安装依赖
请在项目根目录直接执行以下命令安装依赖包，尽管如此仍有可能存在未安装的依赖包，请自行安装。

`pip install -r requirements.txt`

### 1.5 项目结构
```plaintext
sql-focus-finetune/
│
├── README.md           # 项目介绍文档
├── requirements.txt    # 依赖包列表
├── finetune.py         # 微调主程序
├── app/                # Flask框架应用
├── config/             # 配置文件
├── data/               # 数据集
├── model/              # 基础模型文件
├── MSchema/            # M-Schema工具
├── new-model/          # 合并后的模型文件
├── output/             # 微调阶段产生的LoRA权重文件
├── save-model/         # 微调完成后保存的LoRA权重文件
└── utils/              # 数据及模型处理工具
```

## 二、快速开始

### 2.1 指定参数
微调模型时，需要设置多个重要参数,包括模型路径相关参数、数据集路径、训练超参数、LoRA 特定参数等。

**模型路径相关参数**
```angular2html
parser = argparse.ArgumentParser(description="LoRA fine-tuning for model")
    # 模型路径相关参数
parser.add_argument("--model_name", type=str, default="XGenerationLab/XiYanSQL-QwenCoder-7B-2502",
                    help="Path to the model directory downloaded locally")
parser.add_argument("--output_dir", type=str, default="./output",
                    help="Directory to save the fine-tuned model and checkpoints")
```

**数据集路径**
```angular2html
parser.add_argument("--train_file", type=str, default="./data/train_data.jsonl",
                        help="Path to the training data file in JSONL format")
```
**训练超参数**

```angular2html
parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for the input")
parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                    help="Batch size per device during training")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                    help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Number of updates steps to accumulate before performing a backward/update pass")

parser.add_argument("--logging_steps", type=int, default=10,
                    help="Number of steps between logging metrics")
parser.add_argument("--num_train_epochs", type=int, default=3,
                    help="Number of training epochs")
parser.add_argument("--save_steps", type=int, default=500,
                    help="Number of steps between saving checkpoints")
parser.add_argument("--learning_rate", type=float, default=2e-4,
                    help="Learning rate for the optimizer")
```

**--max_seq_length**：最大截断长度，当输入序列超过该长度是做阶段处理。

**--per_device_train_batch_size**：每个设备上的训练批次大小,默认为8。批次大小决定了每次训练时喂给模型的数据量。批次太小可能导致训练过程不稳定或效率低下，批次太大会增加显存占用，可能导致OOM（内存溢出）。

**--per_device_eval_batch_size**：每个设备上的评估批次大小,默认为8。

**--gradient_accumulation_steps**：在执行反向传播/更新操作之前，累积梯度的更新步骤数，默认为1。global batch=num_gpus * per_device_train_batch_size * gradient_accumulation_steps。

- 如果该参数设置的太高的话，会导致梯度累积过多，从而影响模型的学习效率和稳定性，因为梯度是在多个小批量上累积的，而不是每个小批量更新一次，这会导致梯度估计的方差增加，影响模型的收敛性能。
- 另一方面，如果该参数设置的过低的话虽然可以减少梯度累积带来的方差，但相当于减小了有效批量大小，这可能会降低模型训练的效果，因为大批量训练通常能提供更稳定的梯度估计。

**--logging_steps**：每隔多少步记录一次训练日志。不要设置太高，swanlab可能会由于长时间记录不上而导致中断。

**--num_train_epochs**：执行的总训练轮次（如果不是整数，将在停止训练之前执行最后一轮的百分比小数部分）,默认为3.0。

**--save_steps**：每隔多少步保存一次模型，默认为500。

**--learning_rate**：学习率。学习率过高可能会引发梯度爆炸，导致数值溢出，影响模型稳定性。学习率过低则可能导致模型陷入局部最优解，而不是全局最优解。因此我们通常需要通过调参来找到合适的学习率。


**LoRA 特定参数**
```angular2html
parser.add_argument("--lora_rank", type=int, default=16,
                        help="Rank of LoRA matrices")
parser.add_argument("--lora_alpha", type=int, default=32,
                    help="Alpha parameter for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.05,
                    help="Dropout rate for LoRA")
```

**--lora_rank**：LoRA矩阵的秩。

- 较高的`lora_rank`会导致更多的参数需要训练，从而可能提升模型的表示能力，但也会增加训练开销。
- 较低的`lora_rank`则可能降低训练成本，但也可能限制模型的适应能力，导致模型的表现下降。

**--lora_alpha**：LoRA的缩放因子。LoRA矩阵的秩lora_rank通常乘以一个alpha因子进行缩放，这个参数控制低秩矩阵的影响力度。

- `lora_alpha`较大时，LoRA矩阵的影响较大，模型可能会更多地依赖LoRA进行适应，从而影响性能。
- `lora_alpha`较小时，LoRA矩阵的贡献较小，更多地依赖原始模型参数进行预测。选择合适的`lora_alpha`有助于平衡LoRA适应性和训练效率。

**--lora_dropout**：LoRA矩阵中的dropout率。
- 较高的`lora_dropout`值会增加正则化的效果，防止LoRA矩阵过拟合。
- 较低的`lora_dropout`值则可能导致LoRA矩阵过拟合，尤其是在训练数据较少的情况下。
- 对于大多数任务，0.2-0.3 是比较常见地取值，较低的`lora_dropout`值（如 0.1）适合于较小的模型，而较高的`lora_dropout`值（如 0.4-0.5）适合于较大的模型，尤其是在防止过拟合时。

### 2.2 登录Swanlab

这里使用SwanLab来监控整个训练过程，并评估最终的模型效果,初次使用需要[登录SwanLab](https://docs.swanlab.cn/guide_cloud/general/quick-start.html)，更多用法可以参考[官方文档](https://docs.swanlab.cn/)。

#### 安装SwanLab 
使用 pip 在Python3环境的计算机上安装swanlab库。

打开命令行，输入：

`pip install swanlab`

按下回车，等待片刻完成安装。

> 如果遇到安装速度慢的问题，可以指定国内源安装：<br>
`pip install swanlab -i https://mirrors.cernet.edu.cn/pypi/web/simple`

#### 登录账号 
>如果你还没有SwanLab账号，请在[官网](https://swanlab.cn/)免费注册。

打开命令行，输入：

`swanlab login`

当你看到如下提示时：

```angular2html
swanlab: Logging into swanlab cloud.
swanlab: You can find your API key at: https://swanlab.cn/settings
swanlab: Paste an API key from your profile and hit enter, or press 'CTRL-C' to quit:
```

在[用户设置](https://swanlab.cn/settings)页面复制你的 **API Key**，粘贴后按下回车（你不会看到粘贴后的API Key，请放心这是正常的），即可完成登录。之后无需再次登录。

### 2.3 训练模型
在项目根目录(**sql-focus-finetune/**)下，运行以下命令开启训练任务：

`python finetune.py`

大模型微调程序启动后，你可随时前往[Swanlab工作区](https://swanlab.cn/space/~)查看训练过程,训练过程各阶段产生的的权重文件保存在**output**目录下。

### 2.4 合并权重

训练结束后,在**save-model**目录下将得到保存后的权重文件，可使用**utils**目录中的工具合并权重至基础模型，使用方法如下：
```angular2html
cd ./utils
python model_merge.py
```

合并后将得到最终微调后的模型，存放在**new_model**目录下。

## 三、模型应用

本项目结合Flask框架搭建了简易的Text2sql模型应用，你可以通过以下步骤来体验模型的应用：

### 3.1 开启web服务

进入项目根目录，运行以下命令启用web服务：

`python -m app`

也可以指定端口号和主机地址：
- **--port**：指定端口号。
- **--host**：指定主机地址。默认为localhost，即本机地址。

### 3.2 访问应用

开启web服务之后，可使用Postman或其他工具访问api接口，请求地址为：`http://localhost:4060/text2sql`, 请求方法为POST，请求体为json格式，包含以下参数：

```json
{
    "question": "",
    "db_schema": ""
}
```

这里提供一个请求体示例：
```json
{
    "question": "统计2024年1月份的第二产值、第一产值、第三产值",
    "db_schema": "CREATE TABLE \"bi\".\"bi_industry_statistics\" ( \"id\" varchar(32) COLLATE \"pg_catalog\".\"default\" NOT NULL DEFAULT nextval('\"bi\".bi_industry_statistics_id_seg'::regclass), \"tenant_id\" varchar(32) COLLATE \"pg_catalog\".\"default\",\"statistics_time\" timestamp(6), \"province_code\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"city_code\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"county_code\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"town_code\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"village_code\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"statistics_year\" varchar(4) COLLATE \"pg_catalog\".\"default\", \"first_industry\" numeric(20,2), \"second_industry\" numeric(28,2),\"third_industry\" numeric(20,2),\"create_time\" timestamp(6), \"creater\" varchar(32) COLLATE \"pg_catalog\".\"default\", \"update_time\" timestamp(6), \"updater\" varchar(32) COLLATE \"pg_catalog\".\"default\", CONSTRAINT \"bi_industry_statistics_pkey\" PRIMARY KEY (\"id\")); ALTER TABLE \"bi\".\"bi_industry_statistics\" OWNER TO \"topeak_dev\"; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"id\" IS '主键id'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"tenant_id\" IS '租户id'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"statistics_time\" IS '统计时间'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"province_code\" IS '省编码'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"city_code\" IS '市编码'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"county_code\" IS '县编码'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"town_code\" IS '镇编码'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"village code\" IS '村编码'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"statistics_year\" IS '统计年份'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"first_industry\" IS '第一产业产值(单位:元)'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"second_industry\" IS '第二产业产值(单位:元)': COMNENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"third_industry\" IS '第三产业产值(单位:元)'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"create_time\" Is '创建时间'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"creater\" IS '创建人'; COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"update_time\" IS '修改时间';COMMENT ON COLUMN \"bi\".\"bi_industry_statistics\".\"updater\" IS '修改人'; COMMENT ON TABLE \"bi\".\"bi_industry_statistics\" IS '农产统计表';"
}
```

### 3.3 连接数据库

在应用中，你可以使用数据库中的数据生成`db_schema`,而不必每次提交。同时借助开源项目[XGenerationLab/M-Schema](https://github.com/XGenerationLab/M-Schema)提供的工具，可将DDL Schema转化为M-Schema格式，使模型的性能和准确率进一步提高。

- 在使用之前需要先配置数据库连接信息，进入`config`目录，打开`config.ini`文件，修改连接信息：
```ini
[database]
;数据库主机地址
host=127.0.0.1 
;数据库端口
port=3306  
;数据库用户名
user=test
;数据库密码
password=123456
;数据库名称
dbname=example_database
```

**直连数据库开启方式**：启动Flask应用时执行以下命令即可开启。

`python -m app --enable-mschema`

这之后请求体调整为（需重新启动web服务）：
```json
{
    "question": ""
}
```

## 四、模型量化(可选)
如果低参数量的模型输出效果不满足需求，而硬件资源又不足以部署更高参数量的模型，你可以尝试使用模型量化来降低模型的硬件资源消耗，从而在有限硬件资源上部署更大的模型，还能加快模型的推理速度，然而这会使模型输出的准确率下降，但差值仍在可接受范围内，如果量化方法选择得当，差距可进一步缩小，这里介绍一款量化工具：[llama.cpp](https://github.com/ggml-org/llama.cpp)
>`llama.cpp` 的主要目标是使LLM推理能够以最小的配置和更好的性能在更广泛的硬件上运行——无论是在本地还是云端。

在使用此工具之前，需要先将`llama.cpp`下载至本地并进行编译，具体过程可参考[官方提供的教程](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

### 4.1 转换HF模型为GGUF文件

进入`llama.cpp`目录,运行以下命令将HF模型转换为GGUF文件：
```bash
python convert_hf_to_gguf.py XiYanSQL-QwenCoder-7B-R1/ --outfile XiYanSQL-QwenCoder-7B-R1-fp16.gguf
```
>这里对基础模型和输出模型的路径进行了简化处理，请在使用时根据实际情况修改，建议使用绝对路径。

### 4.2 GGUF模型量化
`llama.cpp`提供了多种量化方法：Q2_K、Q3_K_M、Q4_0、Q4_K_S、Q4_K_M、Q5_K_S、Q5_K_M、Q6_K、Q8_0等，其中Q4_K_M最为常用，兼顾性能和资源消耗，因此本项目使用的量化方法也是Q4_K_M，模型量化命令如下：
```bash
./build/bin/llama-quantize XiYanSQL-QwenCoder-7B-R1-fp16.gguf XiYanSQL-QwenCoder-7B-R1-Q4_K_M.gguf Q4_K_M
```

>这里对基础模型和输出模型的路径进行了简化处理，请在使用时根据实际情况修改，建议使用绝对路径。

### 4.3 模型部署
至此你已得到原始的`fp16 GGUF模型文件`和量化后的`Q4_K_M GGUF模型文件`，你可以在[Ollama](https://ollama.com/)、[LMStudio](https://lmstudio.ai/)等本地化部署工具上快速部署模型。

## 五、感谢

本项目的创建是在众多开源项目的基础之上开展的，以下是本项目所依赖的开源项目：
* [Spider](https://github.com/ElementAI/spider)
* [M-Schema](https://github.com/XGenerationLab/M-Schema)
* [deepseek-finetune-lora](https://swanhub.co/Tina_xd/deepseek-finetune-lora)
* [LLM-Finetune](https://github.com/Zeyi-Lin/LLM-Finetune)
* [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master)
* [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub)