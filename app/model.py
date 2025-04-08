from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


class Model:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
            cls._instance.model_name = "./model/XGenerationLab/XiYanSQL-QwenCoder-7B-2502"
            cls._instance.model = AutoModelForCausalLM.from_pretrained(
                cls._instance.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(cls._instance.model_name)
        return cls._instance

    def generate_sql(self, question, db_schema, evidence, dialect="MySQL"):
        nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}
【数据库schema】
{db_schema}
【参考信息】
{evidence}
【用户问题】
{question}
```sql"""

        prompt = nl2sqlite_template_cn.format(dialect=dialect, db_schema=db_schema, question=question,
                                              evidence=evidence)
        message = [{'role': 'user', 'content': prompt}]

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            do_sample=True,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response