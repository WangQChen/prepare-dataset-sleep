import xml.etree.ElementTree as ET
import pandas as pd
import os

def parse_stages(root, path, ns):
    stage_elements = root.findall(path, namespaces=ns)
    stages = []
    for i in range(len(stage_elements) - 1):
        stage_type = stage_elements[i].attrib['Type']
        start = float(stage_elements[i].attrib['Start'])
        end = float(stage_elements[i + 1].attrib['Start'])
        stages.append({'Stage': stage_type, 'Start': start, 'End': end})
    if stage_elements:
        last = stage_elements[-1]
        last_start = float(last.attrib['Start'])
        last_stage = last.attrib['Type']
        stages.append({'Stage': last_stage, 'Start': last_start, 'End': last_start + 30})
    return stages

def generate_epoch_staging(stage_list, gender, epoch_sec=30):
    max_time = max(stage['End'] for stage in stage_list)
    num_epochs = int(max_time // epoch_sec)
    epochs = []
    for i in range(num_epochs):
        start = i * epoch_sec
        end = start + epoch_sec
        stage = next((s['Stage'] for s in stage_list if s['Start'] <= start < s['End']), 'Unknown')
        epochs.append({'epoch': i, 'start': start, 'end': end, 'stage': stage, 'gender': gender})
    return pd.DataFrame(epochs)

def extract_gender(root, ns):
    gender_element = root.find('.//ns:Patient/ns:Gender', namespaces=ns)
    return gender_element.text if gender_element is not None else 'Unknown'

def rml_to_epoch_csv(rml_path, from_machine=False):
    tree = ET.parse(rml_path)
    root = tree.getroot()
    ns = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}

    gender = extract_gender(root, ns)

    if from_machine:
        path = './/ns:StagingData/ns:MachineStaging/ns:NeuroAdultAASMStaging/ns:Stage'
        suffix = '_machine.csv'
    else:
        path = './/ns:StagingData/ns:UserStaging/ns:NeuroAdultAASMStaging/ns:Stage'
        suffix = '_user.csv'

    stage_list = parse_stages(root, path, ns)
    df = generate_epoch_staging(stage_list, gender)

    base_name = os.path.basename(rml_path)
    file_stem = os.path.splitext(base_name)[0]

    output_dir = os.path.join('.', 'data', 'csv', file_stem)  # 修改后的目录结构
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_stem + suffix)

    df.to_csv(output_path, index=False)
    print(f"✅ 生成 {len(df)} 条记录 -> {output_path}，性别: {gender}")

if __name__ == "__main__":
    rml_to_epoch_csv("./data/rml/00000995-100507.rml", from_machine=False)
    rml_to_epoch_csv("./data/rml/00000995-100507.rml", from_machine=True)
