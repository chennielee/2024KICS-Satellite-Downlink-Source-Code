import csv

def compute_aoi_means(csv_file_path):
    """
    aoi_gs.csv 파일을 읽어서,
    aoi_gs_1 ~ aoi_gs_9 열 각각에 대한 평균값을 계산하고 출력한다.
    (aoi_gs_10, aoi_gs_11, aoi_gs_12는 값이 비어있다고 가정하여 무시)
    """
    # 각 aoi_gs_i의 합계(sums)와 개수(counts)를 저장할 리스트
    sums = [0.0] * 10   # 인덱스 0..9 → aoi_gs_0..9
    counts = [0] * 10

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더(컬럼명) 행 건너뛰기

        for row in reader:
            # 빈 줄 등 방어 코드
            if not row:
                continue

            # row[4]가 aoi_gs_0, row[5]가 aoi_gs_1, ... row[13]가 aoi_gs_9
            # row[14], row[15], row[16] = aoi_gs_10..12 (비어있다고 가정)
            
            # 여기서는 aoi_gs_1 ~ aoi_gs_9만 처리
            for i in range(10):
                # 실제 CSV상의 칼럼 인덱스: 4 + i
                val_str = row[4 + i].strip()
                if val_str == "":  # 공백이면 스킵
                    continue
                val = float(val_str)  # 문자열을 float 변환
                sums[i] += val
                counts[i] += 1

    # 계산된 합계와 개수를 이용해 평균 계산
    for i in range(10):
        if counts[i] > 0:
            mean_val = sums[i] / counts[i]
            print(f"aoi_gs_{i} 평균: {mean_val:.2f}")
        else:
            print(f"aoi_gs_{i} 평균: 데이터 없음")

# 예시 사용
if __name__ == "__main__":
    csv_file = "random_steps_log.csv" # aoi_gs.csv
    compute_aoi_means(csv_file)
