import Portfolio_Class as PF

test = PF.Portfolio(["SPY","LQD"],[60,40],1000)
print(test.get_value())
print(test.get_asset_alloc())
for i in range(1000):
	if i % 30 == 0:
		test.rebalance()
	test.update_next()
print(test.get_value())
print(test.get_asset_alloc())
test.histogram(50)