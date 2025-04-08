import json
from tqdm import tqdm

def data_process(data_path,output_path):
    # 最终结果
    all_results = []
    with open(data_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
        instruction = 'I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\"\n##Instruction:\ndepartment_management contains tables such as department, head, management. Table department has columns such as Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees. Department_ID is the primary key.\nTable head has columns such as head_ID, name, born_state, age. head_ID is the primary key.\nTable management has columns such as department_ID, head_ID, temporary_acting. department_ID is the primary key.\nThe head_ID of management is the foreign key of head_ID of head.\nThe department_ID of management is the foreign key of Department_ID of department.\n\n'
        for data in data_json:
            result = {
                'system': instruction,
                'user': data.get('input', ''),
                'assistant': data.get('output', '')
            }
            all_results.append(result)
    # 写入到输出文件中
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in tqdm(all_results, desc="Writing to File"):
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"将 {len(all_results)} 条数据保存到{output_path}中")

if __name__ == '__main__':
    data_path = '../data/original-data/example_text2sql_train.json'
    output_path = '../data/train_data.jsonl'
    data_process(data_path, output_path)
