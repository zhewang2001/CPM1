import json

data = json.load(open('generated_query.json', 'r', encoding='utf8'))

json.dump(data, open('generated_query_new.json', 'w', encoding='utf8'), indent = 4, ensure_ascii=False, )