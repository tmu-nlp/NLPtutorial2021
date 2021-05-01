# マップ(pythonで「辞書」)――疎行列に適している。
#　キーはなんでもOK。次はキーalan 値22からなる辞書の作成

my_dict = {'alan': 22, 'bill': 45, 'chris':17, 'dan':27}

my_dict['eric'] = 33  # 要素の追加
my_dict['alan'] = 23  # 要素の値を変更

print(len(my_dict))
print(my_dict['eric'])   # 要素の値を索引

if 'dan' in my_dict:
    print('dan exists in my_dict')


print(my_dict.items())
# 运行结果：dict_items([('alan', 22), ('bill', 45), ('chris', 17), ('dan', 27), ('eric', 33)])
# 方法.items()返回一个review对象，返回包含每个键值对的元组的列表
tep = my_dict.items()
print(type(tep))

for foo, bar in sorted(my_dict.items()):
    print('%s --> %r' % (foo, bar))
