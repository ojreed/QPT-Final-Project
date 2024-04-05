import Portfolio_Class as PF

Start_Date = (8,8,2013)
Years = 5

Asset_list = ['XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB']
Initial_allocation = [100/len(Asset_list)]*len(Asset_list)
test = PF.Portfolio(asset_list=Asset_list,Investments=Initial_allocation)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
test.smooth_ts(252,True)
# print(test.start)
# while not test.is_done():
# 	if test.is_first_of("month"):
# 		test.rebalance()
# 	test.update_next()
# print(test.get_value())
# print(test.get_asset_alloc())
# test.histogram(60)
