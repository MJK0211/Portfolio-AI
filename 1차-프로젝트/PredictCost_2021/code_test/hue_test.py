import numpy as np

#주식 시장에서 제공하는 한국 증권 휴무일
#2011-12-31일 데이터, 2012-05-01, 2015-08-14


k_stock = np.array(['2012-10-01','2012-10-03','2012-12-25','2012-12-31','2013-01-01','2013-03-01','2013-05-01','2013-05-05','2013-05-17','2013-06-06',
                    '2013-08-15','2013-08-18','2013-08-19','2013-08-20','2013-10-03','2013-10-09','2013-12-25','2013-12-31','2014-01-01','2014-01-30',
                    '2014-01-31','2014-03-01','2014-05-01','2014-05-05','2014-05-06','2014-06-04','2014-06-06','2014-08-15','2014-10-03','2014-10-09',
                    '2014-12-25','2014-12-31','2015-01-01','2015-02-18','2015-02-19','2015-02-20','2015-03-01','2015-05-01','2015-05-05','2015-05-25',
                    '2015-06-06','2015-08-15','2015-09-26','2015-09-27','2015-09-28','2015-09-29','2015-10-03','2015-10-09','2015-12-25','2015-12-31',
                    '2016-01-01','2016-02-08','2016-02-09','2016-02-10','2016-03-01','2016-04-13','2016-05-05','2016-05-06','2016-06-06','2016-08-15',
                    '2016-09-14','2016-09-15','2016-09-16','2016-10-03','2016-12-30','2017-01-27','2017-01-28','2017-01-29','2017-01-30','2017-03-01',	
                    '2017-05-01','2017-05-03','2017-05-05','2017-05-09','2017-06-06','2017-08-15','2017-10-02','2017-10-03','2017-10-04','2017-10-05',	
                    '2017-10-06','2017-10-09','2017-12-25','2017-12-29','2018-01-01','2018-02-15','2018-02-16','2018-03-01','2018-05-01','2018-05-07',
                    '2018-05-22','2018-06-06','2018-06-13','2018-08-15','2018-09-24','2018-09-25','2018-09-26','2018-10-03','2018-10-09','2018-12-25',	
                    '2018-12-31','2019-01-01','2019-02-04','2019-02-05','2019-02-06','2019-03-01','2019-05-01','2019-05-06','2019-06-06','2019-08-15',	
                    '2019-09-12','2019-09-13','2019-10-03','2019-10-09','2019-12-25','2019-12-31','2020-01-01','2020-01-24','2020-01-27','2020-04-15',
                    '2020-04-30','2020-05-01','2020-05-05','2020-08-17','2020-09-30','2020-10-01','2020-10-02','2020-10-09'])

np.save('./project/data/npy/holiday_k_stock.npy', arr=k_stock)

check_holiday = np.array(['2011-12-30', '2012-05-01', '2015-08-14'])
# 2011-12-30 - 12월 30일은 그레고리력으로 364번째(윤년일 경우 365번째) 날
# 2012-05-01 - 2012년 노동절
# 2015-08-14 - 임시 공휴일
# print(k_stock.shape)

np.save('./project/data/npy/check_holiday.npy', arr=check_holiday)