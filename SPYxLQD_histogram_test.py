import Portfolio_Class as PF

Start_Date = (8,8,2013)
Years = 5


test = PF.Portfolio(["SPY","LQD"],[60,40],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
print(test.start)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)

test = PF.Portfolio(["SPY","LQD"],[50,50],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)

test = PF.Portfolio(["SPY","LQD"],[80,20],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)

test = PF.Portfolio(["SPY"],[100],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)

test = PF.Portfolio(["LQD"],[100],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(Start_Date[0],Start_Date[1],Start_Date[2])
test.set_end_relative(Years)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)