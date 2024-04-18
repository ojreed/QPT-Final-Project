import Test_Class as TC

TC = TC.Test_Class((8,8,2018), 5,1)

# TC.get_betas()

TC.test_fixed_alloc("100 SPY",["SPY"],[100])

TC.test_fixed_alloc("100 LQD",["LQD"],[100])

TC.test_fixed_alloc("80-20 SPYxLQD - Daily",["SPY","LQD"],[80,20],0)

TC.test_fixed_alloc("60-40 SPYxLQD - Daily",["SPY","LQD"],[60,40],0)

TC.test_fixed_alloc("50-50 SPYxLQD - Daily",["SPY","LQD"],[50,50],0)

TC.test_fixed_alloc("80-20 SPYxLQD - Monthly",["SPY","LQD"],[80,20],1)

TC.test_fixed_alloc("60-40 SPYxLQD - Monthly",["SPY","LQD"],[60,40],1)

TC.test_fixed_alloc("50-50 SPYxLQD - Monthly",["SPY","LQD"],[50,50],1)

TC.test_fixed_alloc("80-20 SPYxLQD - Yearly",["SPY","LQD"],[80,20],2)

TC.test_fixed_alloc("60-40 SPYxLQD - Yearly",["SPY","LQD"],[60,40],2)

TC.test_fixed_alloc("50-50 SPYxLQD - Yearly",["SPY","LQD"],[50,50],2)

# #full sector allocation
Asset_list = ['XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB']
Initial_allocation = [100/len(Asset_list)]*len(Asset_list)

TC.test_fast_algo("Fast Algo - Daily",Asset_list,Initial_allocation,0)

TC.test_fast_algo("Fast Algo - Monthly",Asset_list,Initial_allocation,1)

TC.test_fast_algo("Fast Algo - Yearly",Asset_list,Initial_allocation,2)

TC.test_RR("Round Robin - Daily",Asset_list,Initial_allocation,15,0.95,0)




TC.plot()
# TC.save("backtest1.csv")