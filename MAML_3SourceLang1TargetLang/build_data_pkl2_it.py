# 訓練用のデータファイルを作る。
import pickle
import sys
import random

#トークナイザー
def tokenizer(sent, word_to_id):
    tokens = sent.lower().split()
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章を単語IDのリスト(tokens_id)に変換
    tokens_ext = ['<sos>'] + tokens + ['<eos>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])
    
    return tokens_id

#ソースとターゲットの文章を読む。
def read_data(lang):

    # ソース言語のファイル名
    file_s = "../MAML_machine_translation_nar_MuTok29/corpus/europarl-v7." + lang + "-en." + lang
    # ターゲット言語（英語）のファイル名
    file_t = "../MAML_machine_translation_nar_MuTok29/corpus/europarl-v7." + lang + "-en.en" 
    
    source = []
    target = []
    with open( file_s, "r", encoding="UTF-8" ) as f_s:
        with open( file_t, "r", encoding="UTF-8" ) as f_t: 
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


def main(sentence_max):
    #メタ学習する言語
    #m_langs = [ "de" ]
    m_langs = [ "it" ]
    #m_langs = [ "de", "fr" ]
    #m_langs = ["de", "fr", "it", "nl", "sv" ]
    #m_langs = ["bg"ブルガリア, "cs"セルビア, "da"デンマーク, "de"ドイツ, "el"ギリシャ, "es"スペイン, "et"エチオピア, "fr", "hu"ハンガリー, "it"イタリア, "lt"リトアニア, "nl"オランダ, "pl"ポーランド, "pt"ポルトガル, "sk"スロバキア, "sl"シエラレオネ, "sv"スウェーデン, "fi"フィンランド, "lv"ラトビア, "ro"ルーマニア ]

    # ファインチューニングする言語
    #f_langs = ["fi", "lv", "ro"]

    # トークナイザーで使う辞書を読み込む。
    with open( "corpus/word_to_id_s.pkl", "rb" ) as f:
        word_to_id_s = pickle.load(f)
    with open( "corpus/word_to_id_t.pkl", "rb" ) as f:
        word_to_id_t = pickle.load(f)

    # reviews に ドメイン（言語の種類）とソース言語のtoken_ids とターゲット言語の token_ids
    reviews = []
    for i, m_lang in enumerate( m_langs ):
        print( "i:", i )
        tmp_s, tmp_t = read_data( m_lang )
        for j, (source, target) in enumerate( zip( tmp_s, tmp_t ) ):
            if j % 10000 == 0:
                print( "j:", j)
            if j > sentence_max:
                break
            r = {}
            r['domain'] = i
            r['source'] = tokenizer(source.lower(), word_to_id_s)
            r['target'] = tokenizer(target.lower(), word_to_id_t)
            reviews.append( r )

    #シャッフル
    random.shuffle(reviews)
    
    point1 = len( reviews ) * 1996 // 2000
    point2 = len( reviews ) * 1998 // 2000
    
    data_train = reviews[:point1]
    data_val = reviews[point1:point2]
    data_test = reviews[point2:]
    
    # ファイルに pkl で書き込み。
    with open("data_train_it.pkl", mode="wb") as f:
        pickle.dump(data_train, f)   
    with open("data_val_it.pkl", mode="wb") as f:
        pickle.dump(data_val, f)   
    with open("data_test_it.pkl", mode="wb") as f:
        pickle.dump(data_test, f)   
     
if __name__ == "__main__":

    # 一言語の文章の最大数。8 * 10000 * 17 で 136万文章。
    #sentence_max = int( sys.argv[1] ) * 10000
    sentence_max = 500000
    
    main( sentence_max )
