import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import os
import datetime
import re
from tqdm._tqdm_notebook import tqdm_notebook
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn import preprocessing

# スクレイピングクラス
class Scraper:
    # レースHTML取得サブルーチン
    @staticmethod
    def scrapeRaceHTML(targetYear):
        # レースID生成サブルーチン
        def generateRaceIdList(targetYear):
            if type(targetYear) != list:
                # リストじゃなかったらリストに変換
                years = [str(targetYear).zfill(4)]
            else:
                # 文字列に変換
                years = [str(y).zfill(4) for y in targetYear]
            # 開催場所 01:札幌 02:函館 03:福島 04:新潟 05:東京
            # 06:中山 07:中京 08:京都 09:阪神 10:小倉
            places = [str(i).zfill(2) for i in range(1, 11)]
            # 開催回(max: 12)
            times = [str(i).zfill(2) for i in range(1, 13)]
            # 開催日(max: 16)
            days = [str(i).zfill(2) for i in range(1, 17)]
            # レースNo(max: 12)
            races = [str(i).zfill(2) for i in range(1, 13)]

            raceIdList = []
            for y in years:
                for p in places:
                    for t in times:
                        for d in days:
                            for r in races:
                                raceIdList.append(y + p + t + d + r)
            return raceIdList

        # 除外リスト生成用関数
        def addEscapeList(id :str, ll :list):
            # raceIdを分解してlist化
            idAry = [id[0:4], id[4:6], id[6:8], id[8:10], id[10:12]]
            holdflg = False
            tescflg = True
            for t in range(int(idAry[2]), 13):
                if holdflg:
                    rtime = 1
                else:
                    # ループ1回目だけ開催日の途中から除外リスト化する
                    rtime = int(idAry[3])
                    holdflg = True
                for d in range(rtime, 17):
                    for r in range(1, 13):
                        raceId = idAry[0] + idAry[1] + \
                                str(t).zfill(2) + str(d).zfill(2) + str(r).zfill(2)
                        ll.append(raceId)
                    if d == 1:
                        tescflg = False
                if tescflg:
                    break
                
            return ll
        
        #-------------------------------------
        # scrapRaceHTMLメイン処理
        #-------------------------------------
        pfx = 'https://db.netkeiba.com/race/'   # URLプレフィックス

        colName = ['raceId', 'htmlBytes']       # 列名をリストで生成
        df = pd.DataFrame(columns=colName)      # データフレームの生成
        escapeList = []
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/race_html.pkl')):
            # pickleファイルを読み込む
            df = pd.read_pickle('./data/race_html.pkl')
            # 読み込んだデータフレームから除外Idリストを生成
            escapeList = df['raceId'].to_list()

        # raceIdリストを生成
        raceIdList = generateRaceIdList(targetYear)
        # 既にescapeListにあるidを除外
        raceIdList = [l for l in raceIdList if l not in escapeList]
        count = 0
        dlist = []
        # raceIdリスト分ループ
        for raceId in tqdm(raceIdList):
            try:
                if raceId in escapeList:        # 除外リストだったらcontinue
                    continue
                ddict = {}
                url = pfx + raceId              # アクセスURLの生成
                html = requests.get(url)        # webアクセスしてHTMLデータを取得
                soup = BeautifulSoup(
                    html.content.decode('euc-jp', 'ignore'), 'html.parser')
                # 結果が存在しているページかチェック
                if '馬名' in soup.text:
                    count += 1
                    # dictにraceIdとhtmlBytesを代入
                    ddict['raceId'] = raceId
                    ddict['htmlBytes'] = html.content
                    # dictをlistに追加
                    dlist.append(ddict)
                else:
                    # 不要なページだったら除外リストを更新
                    escapeList = addEscapeList(raceId, escapeList)

                time.sleep(1)                   # マナーのスレッドスリープ
            except:
                tmpDf = pd.DataFrame(dlist)
                df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
                df = df.sort_values('raceId').reset_index(drop=True)
                df.to_pickle('./data/race_html.pkl')    # データフレームの保存
                raise Exception('例外を検出したので処理を中断しました\n' + 
                                f'{count}件のレース結果を途中保存しました。')

        tmpDf = pd.DataFrame(dlist)
        df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
        df = df.sort_values('raceId').reset_index(drop=True)
        df.to_pickle('./data/race_html.pkl')    # データフレームの保存
        print(f'レース結果のHTMLを{count}件取得しました')
    # レース結果抽出サブルーチン
    @staticmethod
    def extractionRaceResult():
        # レース結果前処理サブルーチン
        def preprocessing(df):
            print('抽出したレース結果のデータを前処理します')
            srcDf = df.copy()        # 加工用にデータフレームをコピー
            # 着外や除外などのデータを欠損データに変換
            srcDf['着順'] = pd.to_numeric(srcDf['着順'], errors='coerce')
            srcDf.dropna(subset=['着順'], inplace=True)
            # 性齢を性と年齢に分割
            srcDf['性'] = srcDf['性齢'].map(lambda x: str(x)[0])
            srcDf['年齢'] = srcDf['性齢'].map(lambda x: str(x)[1:])
            # 馬体重と体重増減を分割して計測不のデータは欠損データに変換
            srcDf['馬体重'] = srcDf['馬体重'].map(lambda x: '---(-)' if '不' in x else x)
            srcDf['体重増減'] = srcDf['馬体重'].str.split('(', expand=True)[1].str[:-1]
            srcDf['馬体重'] = srcDf['馬体重'].str.split('(', expand=True)[0]
            srcDf['馬体重'] = pd.to_numeric(srcDf['馬体重'], errors='coerce')
            srcDf.dropna(subset=['馬体重'], inplace=True)
            srcDf['体重増減'] = pd.to_numeric(srcDf['体重増減'], errors='coerce')
            srcDf.dropna(subset=['体重増減'], inplace=True)
            # 調教師から拠点列を取り出す
            srcDf['拠点'] = srcDf['調教師'].map(lambda x: \
                '東' if '[東]' in x else \
                    '西' if '[西]' in x else \
                        '地' if '[地]' in x else '外')
            # ,を除去後、空の行は0で埋める
            srcDf['賞金'] = srcDf['賞金(万円)'].str.replace(',','')
            srcDf['賞金'] = pd.to_numeric(srcDf['賞金'], errors='coerce')
            srcDf['賞金'] = srcDf['賞金'].fillna(0)

            # 文字列から数値に変換
            for clna in ['着順','年齢','馬体重','体重増減','枠番','馬番']:
                srcDf[clna] = srcDf[clna].astype(int)
            for clna in ['単勝','人気','賞金','斤量']:
                srcDf[clna] = srcDf[clna].astype(float)

            # タイムは扱い易くする為に秒に変換しておく
            srcDf['タイム'] = srcDf['タイム'].map(lambda x: '10:0' if x == '' else x)
            srcDf['タイム'] = (srcDf['タイム']
                .map(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1])))
            srcDf['タイム'] = srcDf['タイム'].map(lambda x: np.nan if x > 550 else x)
            # 着差は1着タイムからの差分を計算して埋める
            srcDf['着差'] = (
                srcDf.groupby('raceId')['タイム'].transform(lambda x: x - x.min()))
            
            # 一応出馬表と同じように馬番で行をソート
            srcDf = srcDf.sort_values(['raceId', '馬番']).reset_index(drop=True)

            # 必要な列だけ拾って、列を並べ替え
            columns = ['raceId', '枠番', '馬番', 'horseId', '馬名', '性', '年齢',
                    '斤量', '騎手', 'jockeyId', '単勝', '人気', 
                    '調教師', 'trainerId', '拠点', '馬体重', '体重増減',
                    '着順', 'タイム', '着差', '通過', '上り', '賞金' ]
            # 並べ替えしたデータフレームを戻り値として返す
            return srcDf[columns].copy()

        #-------------------------------------
        # extractionRaceResultメイン処理
        #-------------------------------------
        # 保存したhtmlデータを読み込む
        htmlDf = pd.read_pickle('./data/race_html.pkl')
        loadDf = pd.DataFrame()
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/race_result.pkl')):
            # pickleファイルを読み込む
            loadDf = pd.read_pickle('./data/race_result.pkl')
            # 読み込んだデータフレームから除外Idリストを生成
            escapeList = loadDf['raceId'].to_list()
            # 未生成のHTMLだけ選別
            htmlDf = htmlDf[~htmlDf['raceId'].isin(escapeList)]

        # 空のデータフレームを生成
        df = pd.DataFrame()
        # 保存したhtmlを1つずつ処理
        for idx, dat in tqdm(htmlDf.iterrows(), total=len(htmlDf)):
            # idとバイナリデータを取り出す
            raceId = dat['raceId']
            htmlBytes = dat['htmlBytes']

            # BeautifulSoupでバイナリデータを解析して0番目のテーブルを取り出す
            soup = BeautifulSoup(htmlBytes.decode('euc-jp', 'ignore'), 'html.parser')
            if '馬名' not in soup.text:
                continue
            table = soup.find_all('table')[0]
            
            # ヘッダー用のList生成してデータフレームを生成
            columns = []
            # thタグを一つずつ取り出してListに追加
            for head in table.find_all('th'):
                columns.append(head.text)
            # 作ったヘッダーListにraceId, horseId, jockeyId, trainerId列を追加
            columns = ['raceId'] + columns + ['horseId', 'jockeyId', 'trainerId']
            oneRaceDf = pd.DataFrame(columns=columns)
            # テーブルを1行毎に処理
            for i, row in enumerate(table.find_all('tr')):
                # 最初の行はヘッダー列なので処理をskip
                if i == 0:
                    continue
                items = [raceId]        # 最初のデータにraceId
                # 1行内のtdを全て取り出す
                cells = row.find_all('td')
                # 1データずつ改行コードを削除しながらデータに追加
                for cell in cells:
                    items.append(cell.text.replace('\n', ''))
                # リンク先を解析しながらhorseId, jockeyId, trainerIdを切り取ってデータに追加
                items.append(str(cells[3]).split('/horse/')[1].split('/')[0])
                items.append(str(cells[6]).split('/recent/')[1].split('/')[0])
                items.append(str(cells[18]).split('/recent/')[1].split('/')[0])
                # 1頭分のデータを追加
                oneRaceDf.loc[i] = items
            # 最後に1レース分のデータフレームを追加
            df = pd.concat([df, oneRaceDf], axis=0)
        # 1件以上のデータを取得していたら前処理を行う
        if len(df) >= 1:
            distDf = preprocessing(df)
            newDf = (pd.concat([loadDf, distDf], axis=0)
                        .sort_values('raceId')
                        .reset_index(drop=True))
            newDf.to_pickle('./data/race_result.pkl')
        print('レース結果のスクレピングが正常に終了しました')
    # レース情報抽出サブルーチン
    @staticmethod
    def extractionRaceInfo():
        # 保存したhtmlデータを読み込む
        htmlDf = pd.read_pickle('./data/race_html.pkl')
        loadDf = pd.DataFrame()
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/race_info.pkl')):
            # pickleファイルを読み込む
            loadDf = pd.read_pickle('./data/race_info.pkl')
            # 読み込んだデータフレームから除外Idリストを生成
            escapeList = loadDf['raceId'].to_list()
            # 未生成のHTMLだけ選別
            htmlDf = htmlDf[~htmlDf['raceId'].isin(escapeList)]

        # 空のリストの生成
        raceInfoList = []

        # 保存したhtmlを1つずつ処理
        for idx, dat in tqdm(htmlDf.iterrows(), total=len(htmlDf)):
            # htmlからraceIdとhtmlBytesを取り出す
            raceId = dat['raceId']
            htmlBytes = dat['htmlBytes']
            # BeautifulSoupでhtmlを解析
            soup = BeautifulSoup(htmlBytes.decode('euc-jp', 'ignore'), 'html.parser')
            if '馬名' not in soup.text:
                continue
            # 情報部分を指定して取り出す
            mainrace_data = soup.find('div', class_='mainrace_data')
            
            # 1レース分のデータを辞書型で宣言
            rowdata = {}
            # raceId
            rowdata['raceId'] = raceId
            # レース名を抽出
            rowdata['レース名'] = mainrace_data.find('h1').text
            # レースNoを抽出
            rowdata['R'] = (mainrace_data.find('dt')
                                .text
                                .replace('\n', '')
                                .replace(' ', '')
                                .replace('R', ''))

            # レース情報部のテキストを取得して'/'で分割
            spantexts = (mainrace_data.find('span').text
                            .replace('\xa0', '')
                            .replace(' ', '')
                            .split('/'))
            # 特定の文字列があるかどうかでコース種を抽出
            rowdata['コース種'] = '障害' if '障' in spantexts[0] else \
                                    'ダート' if 'ダ' in spantexts[0] else '芝'
            # 右・左回りを抽出 ※障害以外は開催場所で決まるからいらないかも？
            rowdata['コース回り'] = '右' if '右' in spantexts[0] else \
                                        '左' if '左' in spantexts[0] else '障害'
            # 距離を抽出
            rowdata['距離'] = int(re.findall('\d+', spantexts[0])[0])
            # 天気、馬場状態を抽出
            rowdata['天気'] = spantexts[1][3:]
            rowdata['馬場'] = spantexts[2].split(':')[1]
            # 発走時間を抽出 ※要らないかも？
            rowdata['発走'] = spantexts[3][3:]
            
            # 次のレース情報部を取得して要らない部分を削除して' 'で分割
            smalltxt = (mainrace_data.find('p', class_='smalltxt')
                            .text
                            .replace('\xa0', ' ')
                            .replace('  ', ' ')
                            .split(' '))
            smalltxtstr = ','.join(smalltxt)
            # 開催日をタイムスタンプに変換し、フォーマットを指定して保存
            dt = datetime.datetime.strptime(smalltxt[0], '%Y年%m月%d日')
            rowdata['日付'] = dt.strftime('%Y/%m/%d')
            # 開催場所はraceIdから判断
            placeDict = {
                '01':'札幌',  '02':'函館',  '03':'福島',  '04':'新潟',  '05':'東京', 
                '06':'中山',  '07':'中京',  '08':'京都',  '09':'阪神',  '10':'小倉'
            }
            rowdata['開催場所'] = placeDict[raceId[4:6]]
            # レースのグレードはまあ色々な方法で判断
            if 'G1' in rowdata['レース名']:
                raceGrade = 'G1'
            elif 'G2' in rowdata['レース名']:
                raceGrade = 'G2'
            elif 'G3' in rowdata['レース名']:
                raceGrade = 'G3'
            elif '未勝利' in smalltxtstr:
                raceGrade = '未勝利'
            elif '新馬' in smalltxtstr:
                raceGrade = '新馬'
            elif '1勝' in smalltxtstr or '500万' in smalltxtstr:
                raceGrade = '1勝クラス'
            elif '2勝' in smalltxtstr or '1000万' in smalltxtstr:
                raceGrade = '2勝クラス'
            elif '3勝' in smalltxtstr or '1600万' in smalltxtstr:
                raceGrade = '3勝クラス'
            else:
                raceGrade = 'オープン'
            rowdata['グレード'] = raceGrade
            # 出走制限を特定の文字列があるかで判断
            if '牡・牝' in smalltxtstr:
                restriction = '牡・牝'
            elif '牝' in smalltxtstr:
                restriction = '牝'
            else:
                restriction = '無'
            rowdata['制限'] = restriction
            # 重量制限を特定の文字列があるかで判断
            if 'ハンデ' in smalltxtstr:
                handicap = 'ハンデ'
            elif '別定' in smalltxtstr:
                handicap = '別定'
            else:
                handicap = '定量'
            rowdata['ハンデ'] = handicap
            # 1レース分のデータをリストに追加
            raceInfoList.append(rowdata)
        # 辞書型をデータフレームに変換
        raceInfoDf = pd.DataFrame(raceInfoList)
        if len(raceInfoDf) >= 1:
            newDf = (pd.concat([loadDf, raceInfoDf], axis=0)
                        .sort_values('raceId')
                        .reset_index(drop=True))
            # データフレームをpickleデータに保存
            newDf.to_pickle('./data/race_info.pkl')
        print('レース情報のスクレピングが正常に終了しました')
    # 血統HTML取得サブルーチン
    @staticmethod
    def scrapePedHTML():
        # レース結果の読込
        raceResultDf = pd.read_pickle('./data/race_result.pkl')

        # URLプレフィックス
        pfx = 'https://db.netkeiba.com/horse/ped/'

        colName = ['horseId', 'htmlBytes']      # 列名をリストで生成
        df = pd.DataFrame(columns=colName)      # データフレームの生成
        escapeList = []
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/ped_html.pkl')):
            # pickleファイルを読み込む
            df = pd.read_pickle('./data/ped_html.pkl')
            # 読み込んだデータフレームから除外Idリストを生成
            escapeList = df['horseId'].to_list()
        
        # raceResultテーブルから未取得のhorseIdを抽出してリストの生成
        horseIdList = (raceResultDf[~raceResultDf['horseId'].isin(escapeList)]['horseId']
                        .unique())

        count = 0
        dlist = []
        # horseIdリスト分ループ
        for horseId in tqdm(horseIdList):
            try:
                if horseId in escapeList:       # 除外リストだったらcontinue
                    continue
                ddict = {}
                url = pfx + horseId             # アクセスURLの生成
                html = requests.get(url)        # webアクセスしてHTMLデータを取得
                soup = BeautifulSoup(html.content, 'html.parser')
                table = soup.find_all('table')[0]
                # 結果が存在しているページかチェック
                if len(table.find_all('a')) != 0:
                    count += 1
                    # horseIdとhtmlBytesを辞書に格納
                    ddict['horseId'] = horseId
                    ddict['htmlBytes'] = html.content
                    dlist.append(ddict)

                time.sleep(1)                   # マナーのスレッドスリープ
            except:
                # データフレームを生成して一時変数に保存
                tmpDf = pd.DataFrame(dlist)
                # データフレームを結合
                df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
                # horseIdでソートしてインデックスの振り直し
                df = df.sort_values('horseId').reset_index(drop=True)
                df.to_pickle('./data/ped_html.pkl')    # データフレームの保存
                raise Exception('例外を検出したので処理を中断しました\n' + 
                                f'{count}件の血統データを途中保存しました')
            
        # データフレームを生成して一時変数に保存
        tmpDf = pd.DataFrame(dlist)
        # データフレームを結合
        df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
        # horseIdでソートしてインデックスの振り直し
        df = df.sort_values('horseId').reset_index(drop=True)
        df.to_pickle('./data/ped_html.pkl')    # データフレームの保存
        print(f'血統データのHTMLを{count}件取得しました')
    # 血統データ抽出サブルーチン
    @staticmethod
    def extractionHorsePed():
        # 保存したhtmlデータを読み込む
        htmlDf = pd.read_pickle('./data/ped_html.pkl')
        # 大丈夫だと思うけど一応インデックスを振り直しておく
        htmlDf = htmlDf.reset_index(drop=True)

        # 血統を取り出す順番をListで定義
        targetList = [0, 31, 1, 16, 32, 47, 2, 9, 17, 24, 33, 40, 48, 55,
                    3, 6, 10, 13, 18, 21, 25, 28, 34, 37, 41, 44, 49, 52, 56, 59,
                    4, 5, 7, 8, 11, 12, 14, 15, 19, 20, 22, 23, 26, 27, 29, 30,
                    35, 36, 38, 39, 42, 43, 45, 46, 50, 51, 53, 54, 57, 58, 60, 61]

        # 列名を生成して
        columns = ['horseId']
        for i in range(62):
            columns.append('pedName_' + str(i))
            columns.append('pedId_' + str(i))
        # 空のデータフレームを生成
        df = pd.DataFrame(columns=columns)

        horsePedDf = pd.DataFrame()
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/horse_ped.pkl')):
            # pickleファイルを読み込む
            horsePedDf = pd.read_pickle('./data/horse_ped.pkl')
            # 読み込んだデータフレームから除外Idリストを生成
            escapeList = horsePedDf['horseId'].to_list()
            # 未生成のHTMLだけ選別
            htmlDf = htmlDf[~htmlDf['horseId'].isin(escapeList)]

        for idx, dat in tqdm(htmlDf.iterrows(), total=len(htmlDf)):
            # htmlからraceIdとhtmlBytesを取り出す
            horseId = dat['horseId']
            htmlBytes = dat['htmlBytes']
            # BeautifulSoupでHTMLを解析
            soup = BeautifulSoup(htmlBytes.decode('euc-jp', 'ignore'), 'html.parser')
            # 血統部分のtdを取り出す
            tds = soup.find_all('table')[0].find_all('td')

            # 最初はhorseId
            rowdata = [horseId]
            for lno in targetList:
                # 名前部分の抽出
                rowdata.append(tds[lno].text.split('\n')[1])
                # IDの抽出
                rowdata.append(str(tds[lno]).split('/horse/')[1].split('/')[0])
            # rowdataをデータフレームに追加
            df.loc[idx] = rowdata
        # 血統データを既存のデータと結合してindexの振り直し
        horsePedDf = (pd.concat([horsePedDf, df], axis=0)
                        .sort_values('horseId')
                        .reset_index(drop=True))
        # データフレームの保存
        horsePedDf.to_pickle('./data/horse_ped.pkl')
        print('血統データのスクレピングが正常に終了しました')
    # 戦績HTML取得サブルーチン
    @staticmethod
    def scrapeHorseResultHTML():
        # レース結果の読込
        raceResultDf = pd.read_pickle('./data/race_result.pkl')
        # 競走馬IDリストの生成
        horseIdList = raceResultDf['horseId'].unique().tolist()

        # スクレイピング対象か確認用にレース情報を読み込む
        raceInfoDf = pd.read_pickle('./data/race_info.pkl')
        # 日付を確認したいからテーブル結合
        raceResultDfM = pd.merge(raceResultDf, raceInfoDf,
                            on='raceId', how='left', suffixes=['', '_right'])

        # URLプレフィックス
        pfx = 'https://db.netkeiba.com/horse/result/'

        colName = ['horseId', 'htmlBytes']      # 列名をリストで生成
        df = pd.DataFrame(columns=colName)      # データフレームの生成
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/horse_html.pkl')):
            # pickleファイルを読み込む
            df = pd.read_pickle('./data/horse_html.pkl')

        # エラー対策をしながら空のデータフレームを生成しておく
        horseResultDf = pd.DataFrame(columns=['horseId', 'raceId', '日付'])
        if(os.path.isfile('./data/horse_result.pkl')):
            # pickleファイルを読み込む
            print('スクレイピング対象となる戦績データを選別します')
            horseResultDf = pd.read_pickle('./data/horse_result.pkl')

            # レース結果と戦績テーブルを日付の降順でソートして必要な部分だけのテーブルにする
            checkDfR = (raceResultDfM[['horseId', '日付']]
                            .sort_values('日付', ascending=False))
            checkDfH = (horseResultDf[['horseId', '日付']]
                            .sort_values('日付', ascending=False))
            # 重複行を削除して直近の日付にする
            checkDfR = checkDfR[~checkDfR['horseId'].duplicated()]
            checkDfH = checkDfH[~checkDfH['horseId'].duplicated()]
            # 二つのテーブルを結合
            checkDfM = pd.merge(left=checkDfR, right=checkDfH,
                                how='left', on='horseId', suffixes=['R', 'H'])
            # 欠損データを埋める
            checkDfM['日付H'] = checkDfM['日付H'].fillna('1000/01/01')
            # 戦績よりレース結果の日付の方が新しいhorseIdだけ抜き取る
            targetList = checkDfM[checkDfM['日付R'] > checkDfM['日付H']]['horseId']
            horseIdList = targetList

        count = 0
        dlist = []
        # horseIdリスト分ループ
        for horseId in tqdm(horseIdList):
            ddict = {}
            # 収集対象となるデータを一旦削除
            df = df[df['horseId'] != horseId]

            try:
                url = pfx + horseId             # アクセスURLの生成
                html = requests.get(url)        # webアクセスしてHTMLデータを取得
                soup = BeautifulSoup(html.content.decode('euc-jp', 'ignore'),
                                        'html.parser')
                # 結果が存在しているページかチェック
                if len(soup.find_all('table')) != 0:
                    count += 1
                    # horseIdとhtmlBytesを辞書型に格納
                    ddict['horseId'] = horseId
                    ddict['htmlBytes'] = html.content
                    # 辞書をlistに追加
                    dlist.append(ddict)

                time.sleep(1)                   # マナーのスレッドスリープ
            except:
                # データフレームを生成して一時変数に保存
                tmpDf = pd.DataFrame(dlist)
                # データフレームを結合
                df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
                # horseIdでソートしてインデックスの振り直し
                df = df.sort_values('horseId').reset_index(drop=True)
                df.to_pickle('./data/horse_html.pkl')    # データフレームの保存
                raise Exception('例外を検出したので処理を中断しました\n' + 
                                f'{count}件の戦績データを途中保存しました')
            
        # データフレームを生成して一時変数に保存
        tmpDf = pd.DataFrame(dlist)
        # データフレームを結合
        df = pd.concat([df, tmpDf], axis=0, ignore_index=True)
        # horseIdでソートしてインデックスの振り直し
        df = df.sort_values('horseId').reset_index(drop=True)
        df.to_pickle('./data/horse_html.pkl')    # データフレームの保存
        print(f'戦績データのHTMLを{count}件取得しました')
    # 戦績抽出サブルーチン
    @staticmethod
    def extractionHorseResult():
        def preprocessing(horseResultDf):
            print('戦績データの前処理を行います')
            df = horseResultDf.copy()
            # 着外や除外などのデータを欠損データに変換
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            df.dropna(subset=['着順'], inplace=True)
            # 馬体重と体重増減を分割して計測不のデータは欠損データに変換
            df['馬体重'] = df['馬体重'].map(lambda x: '---(-)' if '不' in x else x)
            df['体重増減'] = df['馬体重'].str.split('(', expand=True)[1].str[:-1]
            df['馬体重'] = df['馬体重'].str.split('(', expand=True)[0]
            df['馬体重'] = pd.to_numeric(df['馬体重'], errors='coerce')
            df.dropna(subset=['馬体重'], inplace=True)
            df['体重増減'] = pd.to_numeric(df['体重増減'], errors='coerce')
            df.dropna(subset=['体重増減'], inplace=True)
            # ,を除去後、空の行は0で埋める
            df['賞金'] = df['賞金'].astype(str)
            df['賞金'] = df['賞金'].str.replace(',','')
            df['賞金'] = pd.to_numeric(df['賞金'], errors='coerce')
            df['賞金'] = df['賞金'].fillna(0)
            # レース結果と書式を合わせる為に、オッズは単勝に列名を変える
            df['単勝'] = df['オッズ']
            # 変換不可能なデータを欠損データに変換
            df['上り'] = pd.to_numeric(df['上り'], errors='coerce')
            df.dropna(subset=['上り'], inplace=True)

            # 文字列から数値に変換
            for clna in ['R', '頭数', '着順','馬体重','体重増減','枠番','馬番']:
                df[clna] = df[clna].astype(int)
            for clna in ['単勝','人気','賞金','斤量','上り','着差']:
                df[clna] = df[clna].astype(float)

            # タイムは扱い易くする為に秒に変換しておく
            df['タイム'] = df['タイム'].map(lambda x: '10:0' if x == '' else x)
            df['タイム'] = (df['タイム']
                    .map(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1]))
                )   
            df['タイム'] = df['タイム'].map(lambda x: np.nan if x > 550 else x)

            # 日付でソート
            df = df.sort_values('日付')
            # 日付をシリアル値に変換して列を追加
            df['シリアル日'] = (df['日付']
                    .map(lambda x:
                        (datetime.datetime.strptime(x, '%Y/%m/%d') -
                            datetime.datetime(1899, 12, 31)).days + 1)
                )
            # groupbyとdiffでシリアル日の差分を取り出走間隔を演算
            df['出走間隔'] = df.groupby('horseId')['シリアル日'].diff()
            # horseIdと日付でソート
            df = (df.sort_values(['horseId', '日付'], ascending=[True, False])
                    .reset_index(drop=True))
            # 欠損値を0で埋める
            df['出走間隔'] = df['出走間隔'].fillna(0).astype(int)

            # 必要な列だけ拾って、列を並べ替え
            columns = ['horseId', '日付', 'R', 'レース名', 'raceId', '頭数',
                        '枠番', '馬番', '単勝', '人気', '着順', '騎手',
                        'jockeyId', '斤量', 'タイム', '着差', '通過',
                        'ペース', '上り', '馬体重', '体重増減', '出走間隔', '賞金']
            return df[columns].copy()

        #-------------------------------------
        # extractionHorseResultメイン処理
        #-------------------------------------
        # レース結果の読込
        raceResultDf = pd.read_pickle('./data/race_result.pkl')
        # 競走馬IDリストの生成
        horseIdList = raceResultDf['horseId'].unique().tolist()

        # スクレイピング対象か確認用にレース情報を読み込む
        raceInfoDf = pd.read_pickle('./data/race_info.pkl')
        # 日付を確認したいからテーブル結合
        raceResultDfM = pd.merge(raceResultDf, raceInfoDf,
                            on='raceId', how='left', suffixes=['', '_right'])
        # 保存したhtmlデータを読み込む
        htmlDf = pd.read_pickle('./data/horse_html.pkl')
        # 空のデータフレームを生成
        horseResultDf = pd.DataFrame()
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/horse_result.pkl')):
            # pickleファイルを読み込む
            print('抽出対象となる戦績データを選別します')
            horseResultDf = pd.read_pickle('./data/horse_result.pkl')
            def strmerge(x):
                return f'{x["馬体重"]}({x["体重増減"]})'
            horseResultDf['馬体重'] = horseResultDf.apply(strmerge, axis=1)
            def timecalc(x):
                if x != np.nan:
                    itime = int(x * 10)
                    return '{0}:{1}'.format(itime // 600, (itime % 600) * 0.1)
                else:
                    return '10:0'
            horseResultDf['タイム'] = horseResultDf['タイム'].map(timecalc)
            horseResultDf['オッズ'] = horseResultDf['単勝']

            # レース結果と戦績テーブルを日付の降順でソートして必要な部分だけのテーブルにする
            checkDfR = (raceResultDfM[['horseId', '日付']]
                            .sort_values('日付', ascending=False))
            checkDfH = (horseResultDf[['horseId', '日付']]
                            .sort_values('日付', ascending=False))
            # 重複行を削除して直近の日付にする
            checkDfR = checkDfR[~checkDfR['horseId'].duplicated()]
            checkDfH = checkDfH[~checkDfH['horseId'].duplicated()]
            # 二つのテーブルを結合
            checkDfM = pd.merge(left=checkDfR, right=checkDfH,
                                how='left', on='horseId', suffixes=['R', 'H'])
            # 欠損データを埋める
            checkDfM['日付H'] = checkDfM['日付H'].fillna('1000/01/01')
            # 戦績よりレース結果の日付の方が新しいhorseIdだけ抜き取る
            targetList = checkDfM[checkDfM['日付R'] > checkDfM['日付H']]['horseId']

            # 抽出対象となるHTMLだけ選別
            htmlDf = htmlDf[htmlDf['horseId'].isin(targetList)]

        tmpDf = pd.DataFrame()
        # 保存したhtmlを1つずつ処理
        for idx, dat in tqdm(htmlDf.iterrows(), total=len(htmlDf)):
            # idとバイナリデータを取り出す
            horseId = dat['horseId']
            htmlBytes = dat['htmlBytes']

            # BuatifulSoupでHTMLを解析
            soup = BeautifulSoup(htmlBytes.decode('euc-jp', 'ignore'), 'html.parser')
            table = soup.find_all('table')[0]

            # ヘッダー用のList生成してデータフレームを生成
            columns = ['horseId']
            # thタグを一つずつ取り出してListに追加
            for head in table.find_all('th'):
                columns.append(head.text)
            columns += ['raceId', 'jockeyId']
            # 空のデータフレームを生成
            oneHorseDf = pd.DataFrame(columns=columns)
            # 1行ごとに処理を行う
            for i, row in enumerate(table.find_all('tr')):
                if i == 0:
                    continue
                # 競走馬IDを埋め込み
                items = [horseId]
                # cellごとに分解してlistに追加
                for cell in row.find_all('td'):
                    items.append(cell.text.replace('\n', ''))
                # raceIdとjockeyIdを抽出してlistに追加
                items.append(str(row.find_all('td')[4]).split('/race/')[1].split('/')[0])
                try:
                    items.append(
                        str(row.find_all('td')[12]).split('/recent/')[1].split('/')[0])
                except:
                    # たまに騎手にアンカーがない人がいるから例外処理を追加
                    items.append('xxxxx')
                # データフレームに追加
                oneHorseDf.loc[i] = items
            oneHorseDf['馬体重src'] = oneHorseDf['馬体重']
            # データフレームを結合
            tmpDf = pd.concat([tmpDf, oneHorseDf], axis=0)
        # 一時データフレームと本体を結合
        horseResultDf = pd.concat([horseResultDf, tmpDf], axis=0)
        # なんか重複する時があるからhorseIdと日付をキーに重複行削除
        horseResultDf = horseResultDf[~horseResultDf[['horseId', '日付']].duplicated()]
        df = preprocessing(horseResultDf)

        # pickleデータの保存
        df.to_pickle('./data/horse_result.pkl')
        print('戦績データのスクレピングが正常に終了しました')
    # 出馬表スクレイピングサブルーチン
    @staticmethod
    def scrapeEntryTable(raceId):
        def getSoup(raceId):
            pfx = 'https://race.netkeiba.com/race/shutuba.html?race_id='
            url = pfx + raceId

            # seleniumオプションのインスタンス
            options = Options()
            # ウィンドウの非表示を設定
            options.add_argument('--headless')
            # driverをインスタンス
            driver = webdriver.Chrome(
                './tool/chromedriver_win32/chromedriver.exe', options=options)

            # urlからWebページを取得
            driver.get(url)
            # HTMLソースを抽出
            html = driver.page_source
            # driverを閉じる
            driver.close()
            # HTMLソースをBeautifulSoupで解析
            soup = BeautifulSoup(html, 'html.parser', from_encoding='euc-jp')

            return soup
        
        def extractEntryDf(raceId, soup):
            table = soup.find_all('table')[0]
            tbody = table.find('tbody')

            dictList = []
            for idx, tr in enumerate(tbody.find_all('tr')):
                dic = {}
                dic['raceId'] = raceId
                dic['枠番'] = tr.find('td', class_=re.compile('^Waku')).text
                dic['馬番'] = tr.find('td', class_=re.compile('^Umaban')).text
                horseInfo = tr.find('td', class_='HorseInfo')
                dic['馬名'] = horseInfo.text.replace('\n', '')
                try:
                    dic['horseId'] = (str(horseInfo.find('a'))
                                        .split('/horse/')[1].split('"')[0])
                    dic['性'] = tr.find('td', class_='Barei').text[0]
                    dic['年齢'] = tr.find('td', class_='Barei').text[1:]
                    dic['斤量'] = tr.find_all('td', class_='Txt_C')[3].text
                    jockey = tr.find('td', class_='Jockey')
                    anker = jockey.find('a')
                    dic['騎手'] = anker.text
                    dic['jockeyId'] = str(anker).split('/recent/')[1].split('/')[0]
                    trainer = tr.find('td', class_='Trainer')
                    anker = trainer.find('a')
                    dic['調教師'] = anker.text
                    dic['trainerId'] = str(anker).split('/recent/')[1].split('/')[0]
                    span = trainer.find('span')
                    dic['拠点'] = '東' if '美浦' in span.text else \
                                    '西' if '栗東' in span.text else \
                                        '他' if '他' in span.text else '外'
                    dic['単勝'] = tr.find('td', class_='Popular').text
                    dic['人気'] = tr.find('td', class_='Popular_Ninki').find('span').text
                except:
                    pass
                try:
                    weight = tr.find('td', class_='Weight')
                    tmp = weight.text.replace('\n', '').split('(')
                    dic['馬体重'] = tmp[0]
                    num = tmp[1].split(')')[0]
                    dic['体重増減'] = np.nan if '不' in num else num
                except:
                    dic['馬体重'] = np.nan
                    dic['体重増減'] = np.nan
                dictList.append(dic)
            entryDf = pd.DataFrame(dictList)
            return entryDf
        
        def extractInfo(raceId, soup):
            columns = [
                '日付', 'ハンデ', '着順', 'R', 'コース種', 'コース回り', '距離',
                '天気', '馬場', '開催場所', 'グレード', '制限'
            ]

            dic = {}
            dl = soup.find('dl', id='RaceList_DateList')
            md = dl.find('dd', class_='Active').text.split('(')[0]
            year = str(datetime.date.today().year)
            targetDate = year + '年' + md
            dt = datetime.datetime.strptime(targetDate, '%Y年%m月%d日')
            dic['日付'] = dt.strftime('%Y/%m/%d')

            raceData01 = soup.find('div', class_='RaceData01')
            spans1 = raceData01.find_all('span')
            raceData02 = soup.find('div', class_='RaceData02')
            spans2 = raceData02.find_all('span')
            dic['ハンデ'] = 'ハンデ' if 'ハンデ' in spans2[6].text else \
                                '別定' if '別定' in spans2[6].text else '定量'
            div = soup.find('div', class_='RaceList_Item01')
            dic['R'] = int(div.text.replace('\n', '')[:-1])
            dic['コース種'] = '障害' if '障' in spans1[0].text else \
                                'ダート' if 'ダ' in spans1[0].text else '芝'
            dic['コース回り'] = '右' if '右' in raceData01.text else \
                                    '左' if '左' in raceData01.text else '障害'
            dic['距離'] = int(re.findall('\\d+', spans1[0].text)[0])
            try:
                dic['天気'] = raceData01.text.split('天候:')[1].split('\n')[0]
            except:
                wt = ''
                while(True):
                    wt = input('予想天気を入力。晴,曇,小雨,雨,小雪,雪')
                    if wt in ['晴','曇','小雨','雨','小雪','雪']:
                        dic['天気'] = wt
                        break
                    elif wt == '':
                        dic['天気'] = np.nan
                        break
            try:
                dic['馬場'] = raceData01.text.split('馬場:')[1].split('\n')[0]
                dic['馬場'] = '不良' if dic['馬場'] == '不' else dic['馬場']
                dic['馬場'] = '稍重' if dic['馬場'] == '稍' else dic['馬場']
            except:
                gs = ''
                while(True):
                    wt = input('予想馬場を入力。良,稍重,重,不良,良ダート,稍重ダート,重ダート,不良ダート')
                    if wt in ['良','稍重','重','不良',
                            '良ダート', '稍重ダート','重ダート','不良ダート']:
                        dic['馬場'] = wt
                        break
                    elif wt == '':
                        dic['馬場'] = np.nan
                        break
            placeDict = {
                '01':'札幌',  '02':'函館',  '03':'福島',  '04':'新潟',  '05':'東京', 
                '06':'中山',  '07':'中京',  '08':'京都',  '09':'阪神',  '10':'小倉'
            }
            dic['開催場所'] = placeDict[raceId[4:6]]

            if '未勝利' in spans2[4].text:
                raceGrade = '未勝利'
            elif '新馬' in spans2[4].text:
                raceGrade = '新馬'
            elif '1勝' in spans2[4].text or '500万' in spans2[4].text:
                raceGrade = '1勝クラス'
            elif '2勝' in spans2[4].text or '1000万' in spans2[4].text:
                raceGrade = '2勝クラス'
            elif '3勝' in spans2[4].text or '1600万' in spans2[4].text:
                raceGrade = '3勝クラス'
            else:
                raceName = soup.find('div', 'RaceName')
                if 'GradeType1"' in str(raceName):
                    raceGrade = 'G1'
                elif 'GradeType2"' in str(raceName):
                    raceGrade = 'G2'
                elif 'GradeType3"' in str(raceName):
                    raceGrade = 'G3'
                else:
                    raceGrade = 'オープン'

            dic['グレード'] = raceGrade
            if '牡・牝' in spans2[5].text:
                restriction = '牡・牝'
            elif '牝' in spans2[5].text:
                restriction = '牝'
            else:
                restriction = '無'
            dic['制限'] = restriction
            dic['raceId'] = raceId
            raceInfo = pd.DataFrame([dic])
            return raceInfo

        soup = getSoup(raceId)
        entryDf = extractEntryDf(raceId, soup)
        infoDf = extractInfo(raceId, soup)
        entryTable = pd.merge(left=entryDf, right=infoDf, how='left', on='raceId')
        return entryTable, entryDf, infoDf

# データセットクラス
class Dataset:
    # コンストラクタ
    def __init__(self, load=True):
        self.__columnsDict = self.__generateColumns()
        self.__encodeEnvs = self.__generateEncodeEnvs()
        if load:
            self.__dataset = pd.read_pickle('./data/dataset.pkl')
    @property
    def dataset(self):
        return self.__dataset[self.columnsDict['all']].copy()
    @property
    def columnsDict(self):
        return self.__columnsDict.copy()
    # エンコード実行メソッド
    def encoding(self, src, fit=False):
        # データセットをコピー
        df = src.copy()
        for key, env in tqdm(self.__encodeEnvs.items()):
            cols = env['cols']
            le = env['encoder']
            if fit:
                # fitの指示があったらデータフレームから対象の値を取り出して実行
                na = df[cols].to_numpy()
                tg = na.reshape(-1).tolist()
                le.fit(tg)
            for col in cols:
                # 欠損データ以外の列を取り出す
                notNull = df[col][df[col].notnull()]
                # エンコード実行してindexをキーにデータフレームに書き込む
                df[col] = pd.Series(le.transform(notNull), index=notNull.index)
                # エンコードした列はcategory列に変換
                df[col] = df[col].astype('category')
        cols = self.__columnsDict['numeric']
        for col in cols:
            df[col] = df[col].astype(float)

        return df
    # 出馬表用前処理メソッド
    def preprocessingEntryTable(self, entryDf): 
        df = self.addHistrical(entryDf)
        df = self.addInterval(df)
        df = self.addPeds(df)
        df = df[self.__columnsDict['entry_table']]
        df = self.encoding(df)
        return df
    # データセット一括更新メソッド
    def update(self, targetYear):
        print('レース結果のHTMLを取得します')
        Scraper.scrapeRaceHTML(targetYear)
        print('レース結果を抽出します')
        Scraper.extractionRaceResult()
        print('レース情報を抽出します')
        Scraper.extractionRaceInfo()
        print('血統データのHTMLを取得します')
        Scraper.scrapePedHTML()
        print('血統データを抽出します')
        Scraper.extractionHorsePed()
        print('戦績データのHTMLを取得します')
        Scraper.scrapeHorseResultHTML()
        print('戦績データを抽出します')
        Scraper.extractionHorseResult()

        self.__dataset = self.__updateDataset()
    # 列名生成メソッド
    def __generateColumns(self):
        # 使用する列名を指定
        resultCol = [
            '日付', 'raceId', '枠番', '馬番', 'horseId', '性', '年齢',
            '斤量', 'jockeyId', '単勝', '人気', 'trainerId', '拠点',
            '馬体重', '体重増減', '出走間隔', 'ハンデ', '着順', 'R',
            'コース種', 'コース回り', '距離', '天気', '馬場', '開催場所',
            'グレード', '制限', '頭数'
        ]
        recordCol = [
            'R', '頭数', '枠番', '馬番', '単勝', '人気', '着順',
            'jockeyId', '斤量', 'タイム', '着差', '上り', '馬体重',
            '体重増減', '出走間隔', 'コース種', 'コース回り', '距離',
            '天気', '馬場', '開催場所', 'グレード', '制限', 'ハンデ'
        ]
        pedCol = ['pedId_' + str(i) for i in range(0, 62)]
        # 前N走分戦績の列名を生成
        recordCol9 = []
        for i in range(1, 10):
            tmpList = list(map(lambda x: x + '_' + str(i), recordCol))
            recordCol9 += tmpList

        # 生成した列名を辞書に格納
        columnsDict = {}
        columnsDict['all'] = resultCol + recordCol9 + pedCol
        columnsDict['histrical'] = recordCol9
        columnsDict['ped'] = pedCol
        l = columnsDict['all'].copy()
        l.remove('着順')
        columnsDict['entry_table'] = l

        # 量的変数の列名を生成
        numericCols = ['年齢']
        cols1 = ['枠番', '馬番', '単勝', '人気', '斤量',
                '馬体重', '体重増減', '出走間隔', 'R', '距離']
        cols2 = ['頭数', '着順', 'タイム', '着差', '上り']
        numericCols += cols1
        cols3 = cols1 + cols2
        for i in range(1, 10):
            numericCols += map(lambda x: x + '_' + str(i), cols3)
        columnsDict['numeric'] = numericCols

        # カテゴリ変数の列名を生成
        sr = pd.Series(columnsDict['all'])
        categoryCol = sr[~sr.isin(numericCols)].to_list()
        columnsDict['categorical'] = categoryCol

        return columnsDict
    # エンコーダー生成メソッド
    def __generateEncodeEnvs(self):
        encodeEnvs = {}
        # horseIdだけは勝手が違うので個別に環境を生成
        dd = {}
        dd['cols'] = ['horseId'] + [s for s in self.__columnsDict['all'] if 'ped' in s]
        dd['encoder'] = preprocessing.LabelEncoder()
        encodeEnvs['horseId'] = dd
        # それ以外の対象はリスト化してfor文で処理
        cols = ['性', 'trainerId', '拠点', 'jockeyId', 'ハンデ', 'コース種',
                'コース回り', '天気', '馬場', '開催場所', 'グレード', '制限']
        for col in cols:
            dd = {}
            dd['cols'] = [s for s in self.__columnsDict['all'] if col in s]
            dd['encoder'] = preprocessing.LabelEncoder()
            encodeEnvs[col] = dd
        return encodeEnvs
    # 過去データセットの更新
    def __updateDataset(self):
        print('レース結果のデータをロードします')
        # 保存したデータを読み込みレース結果とレース情報を結合
        dataDf = self.__generateRaceResult()
        loadDf = pd.DataFrame()
        # pickleファイルの存在チェック
        if(os.path.isfile('./data/race_result_add_record.pkl')):
            print('処理済みの戦績付きデータを読み込みます')
            # pickleファイルを読み込む
            loadDf = pd.read_pickle('./data/race_result_add_record.pkl')
            print('加工対象データを抽出します')
            # 保存したファイルを読み込んで加工対象のみ絞り込む
            loadDf['key'] = loadDf.apply(lambda x: x['horseId'] + x['日付'], axis=1)
            dataDf['key'] = dataDf.apply(lambda x: x['horseId'] + x['日付'], axis=1)
            dataDf = dataDf[~dataDf['key'].isin(loadDf['key'])]
            dataDf = dataDf.drop('key', axis=1)
            loadDf = loadDf.drop('key', axis=1)
        
        # 加工対象があったら加工実行
        if len(dataDf) > 0:
            print('レース結果に戦績データを付与します')
            # 上記のデータに戦績データを付与
            dataAddHis = Dataset.addHistrical(dataDf)
            if len(loadDf) >= 0:
                dataAddHis = pd.concat([loadDf, dataAddHis], axis=0)
            print('戦績付きデータを保存します')
            dataAddHis.to_pickle('./data/race_result_add_record.pkl')
        else:
            dataAddHis = loadDf
        
        print('出走間隔を計算し列に追加します')
        dataAddInterval = Dataset.addInterval(dataAddHis)
        print('血統データを付与します')
        dataAddPeds = Dataset.addPeds(dataAddInterval)
        dataset = (dataAddPeds
                    .sort_values(['日付', 'raceId', '馬番'], ascending=[True, True, True])
                    .reset_index(drop=True))
        print('データセットをファイルに保存します')
        dataset.to_pickle('./data/dataset.pkl')

        print('データセットの更新が完了しました')
        return dataset
    # レース結果とレース情報をファイルからロード
    def __generateRaceResult(self):
        # pickleファイルからデータのロード
        df = pd.read_pickle('./data/race_result.pkl')
        infoDf = pd.read_pickle('./data/race_info.pkl')
        # レース結果と競走馬戦績にレース情報を結合
        result = pd.merge(left=df, right=infoDf,
                    how='left', on='raceId', suffixes=['', '_i'])
        return result
    # レース結果または出馬表データに戦績データ付与するメソッド
    @staticmethod
    def addHistrical(srcdf):
        tqdm_notebook.pandas()
        # pickleファイルからデータのロード
        dfRaceInfo = pd.read_pickle('./data/race_info.pkl')
        dfHorseResult = pd.read_pickle('./data/horse_result.pkl')
        # 競走馬戦績にレース情報を結合
        dfHorseResulti = pd.merge(left=dfHorseResult, right=dfRaceInfo,
                                how='left', on='raceId', suffixes=['', '_i'])
        # 戦績テーブルからゴミを除去
        df = dfHorseResulti.copy()
        columns = []
        for cn in df.columns:
            if '_i' not in cn:
                columns.append(cn)
        dfHorseResulti = df[columns]
        
        # 引数のデータフレームをコピー
        df = srcdf.copy()
        # 戦績データとする列を_1～_9のサフィックスを付与して一度numpy配列にする
        cols = np.array([dfHorseResulti.add_suffix(f'_{i}').columns for i in range(1, 10)])
        # 列名を1次元配列にreshape
        cols = cols.reshape(-1)

        # 戦績データ生成関数の定義
        def generateHistoricalData(row):
            # 日付とhorseIdの抽出
            dt = row['日付']
            horseId = row['horseId']
            # 戦績データを日付とhorseIdで絞り込み
            na = (
                dfHorseResulti[(dfHorseResulti['日付']<dt)&
                               (dfHorseResulti['horseId']==horseId)]
                    .head(9)        # 直近9レース分に限定
                    .to_numpy()     # numpy配列に変換
                    .reshape(-1)    # 1次元配列に変換
            )
            # 戦績データのサイズが9レース分無かったら足りないサイズ分nanで埋める
            if na.size < cols.size:
                naNan = np.array([np.nan for i in range(cols.size - na.size)])
                na = np.concatenate([na, naNan])
            return na
        # 戦績データを生成してapplyで関数実行
        df[cols] = df.progress_apply(generateHistoricalData, axis=1, result_type='expand')

        return df
    # レース結果または出馬表データの出走間隔を計算するメソッド
    @staticmethod
    def addInterval(srcdf):
        df = srcdf.copy()
        # スカラー関数の定義
        def calcInterval(x):
            try:
                interval = (datetime.datetime.strptime(x['日付'], '%Y/%m/%d') - 
                            datetime.datetime.strptime(x['日付_1'], '%Y/%m/%d'))
                return interval.days
            except:
                return 0
        # apply関数で一括処理
        df['出走間隔'] = df.progress_apply(calcInterval, axis=1)
        return df
    # レース結果または出馬表データに血統データを付与するメソッド
    @staticmethod
    def addPeds(srcdf):
        df = srcdf.copy()
        dfHorsePed = pd.read_pickle('./data/horse_ped.pkl')
        return pd.merge(left=df, right=dfHorsePed, how='left', on='horseId')

def version():
    mejor = 0
    minor = 1
    revsion = 2
    print(f'ゆっくり競馬予想プログラム Ver.{mejor}.{minor}.{revsion}')
    comment = (
'''
Release note
Ver0.1.2
レース結果とレース情報内のスキップ処理のバグを修正
'''
)
    print(comment)

if __name__ == '__main__':
    version()

