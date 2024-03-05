import Portfolio_Class as PF

test = PF.Portfolio(["SPY","LQD"],[60,40],1000)
print(test.get_value())
print(test.get_asset_alloc())
test.set_start(1,2,2009)
test.set_end_relative(5)
print(test.start)
print(test.end)
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
test.set_start(1,2,2009)
test.set_end_relative(5)
print(test.start)
print(test.end)
while not test.is_done():
	if test.is_first_of("month"):
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(60)