import os
import shutil
import glob
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Generate CamelNeon Advertise Play List")
    parser.add_argument("--mode", type=bool, default=True)
    return parser

def restore_files_from_used(used_dir, root_dir):
    # used 디렉토리 내 모든 파일 검색
    all_files = glob.glob(os.path.join(used_dir, '**', '*'), recursive=True)
    
    for file_path in tqdm(all_files):
        # 파일인지 확인
        if os.path.isfile(file_path):
            # used 경로 기준으로 원본 경로 계산
            relative_path = os.path.relpath(file_path, used_dir)
            original_path = os.path.join(root_dir, relative_path)
            
            # 원본 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            
            # 파일 복사 (원상복구)
            shutil.copy2(file_path, original_path)

def copy_non_ipynb_py_files_to_used(root_dir, used_dir):
    # 하위 디렉토리 포함 모든 파일 검색
    all_files = glob.glob(os.path.join(root_dir, '**', '*'), recursive=True)
    
    for file_path in tqdm(all_files):
        # 파일인지 확인
        if os.path.isfile(file_path):
            # 확장자가 .ipynb 또는 .py가 아닌 경우 복사
            if not (file_path.endswith('.ipynb') or file_path.endswith('.py')):
                # 원본 경로 기준으로 used_data 경로 생성
                relative_path = os.path.relpath(file_path, root_dir)
                target_path = os.path.join(used_dir, relative_path)
                
                # 타겟 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # 파일 복사
                shutil.copy2(file_path, target_path)
                
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    if mode:
        restore_files_from_used('./used_data', './')
    else:
        copy_non_ipynb_py_files_to_used('./', './used_data')