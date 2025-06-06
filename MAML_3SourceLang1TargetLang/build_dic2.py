import pickle
#from pycocotools.coco import COCO
from collections import Counter

def read_data(lang):

    file_s = "../MAML_machine_translation_nar_MuTok29/corpus/europarl-v7." + lang + "-en." + lang
    file_t = "../MAML_machine_translation_nar_MuTok29/corpus/europarl-v7." + lang + "-en.en" 
    
    source = []
    target = []
    with open( file_s, "r", encoding="UTF-8" ) as f_s:
        with open( file_t, "r", encoding="UTF-8"  ) as f_t: 
            s_line = f_s.readline()
            t_line = f_t.readline()
            source.append( s_line )
            target.append( t_line )
            while s_line and t_line:
                s_line = f_s.readline()
                t_line = f_t.readline()
                source.append( s_line )
                target.append( t_line )
        
    return source, target

# データの保存先
#m_langs = [ "de" ]
m_langs = [ "de", "fr", "it" ]
#m_langs = ["de", "fr", "it", "nl", "sv" ]
#m_langs = ["bg"ブルガリア, "cs"セルビア, "da"デンマーク, "de"ドイツ, "el"ギリシャ, "es"スペイン, "et"エチオピア, "fr", "hu"ハンガリー, "it"イタリア, "lt"リトアニア, "nl"オランダ, "pl"ポーランド, "pt"ポルトガル, "sk"スロバキア, "sl"シエラレオネ, "sv"スウェーデン, "fi"フィンランド, "lv"ラトビア, "ro"ルーマニア ]
fp_word_to_id_s = 'corpus/word_to_id_s.pkl'
fp_id_to_word_s = 'corpus/id_to_word_s.pkl'
fp_word_to_id_t = 'corpus/word_to_id_t.pkl'
fp_id_to_word_t = 'corpus/id_to_word_t.pkl'


#sentence_max = 80000
# 特殊トークンの追加
#vocab.insert( 0, '<blank>' )
#vocab.append('<sos>') # 文章の始まりを表すトークンを追加
#vocab.append('<eos>')  # 文章の終わりを表すトークンを追加
#vocab.append('<unk>')  # 辞書に無い単語を表すトークンを追加
#vocab.append('<pad>')  # 系列長を揃えるためのトークンを追加
id_to_word_s = {0:'<pad>', 1:'<sos>',2:'<eos>',3:'<unk>',4:'<blank>',5:'<mask>' }
id_to_word_t = {0:'<pad>', 1:'<sos>',2:'<eos>',3:'<unk>',4:'<blank>',5:'<mask>' }
word_to_id_s = {'<pad>':0, '<sos>':1,'<eos>':2,'<unk>':3,'<blank>':4,'<mask>':5 }
word_to_id_t = {'<pad>':0, '<sos>':1,'<eos>':2,'<unk>':3,'<blank>':4,'<mask>':5 }
all_s = {'<blank>':1000, '<sos>':1000,'<eos>':1000,'<unk>':1000,'<pad>':1000,'<mask>':1000 }
all_t = {'<blank>':1000, '<sos>':1000,'<eos>':1000,'<unk>':1000,'<pad>':1000,'<mask>':1000 }
thresh_s = 100
thresh_t = 100
#id_to_word = {}
#word_to_id = {}
for i, m_lang in enumerate( m_langs ):
    print( "i:", i )
    tmp_s, tmp_t = read_data( m_lang )
    for j, (source, target) in enumerate( zip( tmp_s, tmp_t ) ):
        if j % 10000 == 0:
            print( "j:", j)
        #if j > sentence_max:
        #    break
        tokens = source.lower().split()
        #print( "tokens:", tokens )
        #print( "word_to_id:", word_to_id )
        for token in tokens:
            token = token.replace( ".", "" ).replace( ",", "" )
            if token in all_s:
                all_s[token] += 1
            else:
                all_s[token] = 1
            if token not in word_to_id_s and all_s[token] > thresh_s:
                new_id = len( word_to_id_s )
                id_to_word_s[new_id] = token
                word_to_id_s[token] = new_id
        tokens = target.lower().split()
        for token in tokens:
            token = token.replace( ".", "" ).replace( ",", "" )
            if token in all_t:
                all_t[token] += 1
            else:
                all_t[token] = 1
            if token not in word_to_id_t and all_t[token] > thresh_t:
                new_id = len( word_to_id_t )
                id_to_word_t[new_id] = token
                word_to_id_t[token] = new_id
    print(f'単語数_s: {str(len(word_to_id_s))}')    
    print(f'単語数_t: {str(len(word_to_id_t))}')    


# ピリオド、カンマを削除
#table = str.maketrans({'.': '',
#                       ',': ''})
#for k in range(len(coco_token)):
#    coco_token[k] = coco_token[k].translate(table)

# 単語ヒストグラムを作成
#freq = Counter(coco_token)

# 3回以上出現する単語に限定して辞書を作成
#vocab = [token for token, count in freq.items() if count >= 3]
#sorted(vocab)


# 単語ー単語ID対応表の作成
#word_to_id = {token: i for i, token in enumerate(vocab)}
#id_to_word = {i: token for i, token in enumerate(vocab)}

# ファイル出力
with open(fp_word_to_id_s, 'wb') as f:
    pickle.dump(word_to_id_s, f)
with open(fp_id_to_word_s, 'wb') as f:
    pickle.dump(id_to_word_s, f)
with open(fp_word_to_id_t, 'wb') as f:
    pickle.dump(word_to_id_t, f)
with open(fp_id_to_word_t, 'wb') as f:
    pickle.dump(id_to_word_t, f)


print(f'単語数_s: {str(len(word_to_id_s))}')
print(f'単語数_t: {str(len(word_to_id_t))}')
