import pandas as pd
import numpy as np

# TODO: implement test cases
# dict1 = {
#     "feat1": ["metric1", "metric2"],
#     "feat2": ["metric3", "metric4", "metric5"],
#     "feat3": ["metric6"],
#     "feat4": ["metric7"],
#     "feat5": ["metric8"],
#     "feat6": ["metric9"],
# }
# dict2 = {
#     "feat1": ["feat1", "feat2"],
#     "feat2": ["feat3", "feat4"],
#     "feat3": ["feat5"],
#     "feat4": ["feat6"],
# }
# dict3 = {
#     "feat1": ["feat1", "feat2", "feat3"],
#     "feat2": ["feat4"],
# }
# steps = [dict1, dict2, dict3]
# 
# 
# #
# comb = steps[-1]
# for step in steps[:-1][::-1]:
#     comb = sub_values(comb, step)

# # %% [markdown]
# # ## Test CorrFeatureAgglomeration w/ Pipeline
# 
# # %%
# Ximg = X.filter(like="image8_")
# pipe = make_pipeline(
#     SimpleImputer(strategy="mean"),
#     ScaledVarianceThreshold(),
#     CorrFeatureAgglomeration(r=0.8),
# )
# pipe.fit(Ximg)
# 
# # make display
# display = FeatureAgglomerationDisplay(Ximg, pipe)
# plist = []
# plist.append(display.plot_dendrogram())
# plist.append(display.plot_in_corr_map().fig)
# plist.append(display.plot_out_corr_map().fig)
# 
# # set sizes
# # for p, size in zip(plist, [(5, 3), (8, 8), (5, 5)]):
# #     p.set_size_inches(size)
# # plt.show()
# # plt.close()
# 
# # save to pdf
# save_multi_page_pdf(plist, "onestep_w_pipeline.pdf")
# 
# # %% [markdown]
# # ## Test CorrFeatureAgglomeration w/o Pipeline
# 
# # %%
# Ximg = ScaledVarianceThreshold().fit_transform(SimpleImputer().fit_transform(X.filter(like="image8_")))
# pipe = CorrFeatureAgglomeration(r=0.8)
# pipe.fit(Ximg)
# 
# # make display
# display = FeatureAgglomerationDisplay(Ximg, pipe)
# plist = []
# plist.append(display.plot_dendrogram())
# plist.append(display.plot_in_corr_map().fig)
# plist.append(display.plot_out_corr_map().fig)
# 
# # set sizes
# # for p, size in zip(plist, [(5, 3), (8, 8), (5, 5)]):
# #     p.set_size_inches(size)
# # plt.show()
# # plt.close()
# 
# # save to pdf
# save_multi_page_pdf(plist, "onestep_no_pipeline.pdf")
# 
# # %% [markdown]
# # ## Test IterativeCorrFeatureAgglomeration with Pipeline
# 
# # %%
# # test for IterativeCorrFeatureAgglomeration
# Ximg = X.filter(like="image8_")
# pipe = make_pipeline(
#     SimpleImputer(strategy="mean"),
#     ScaledVarianceThreshold(),
#     IterativeCorrFeatureAgglomeration(r=0.8),
# )
# pipe.fit(Ximg)
# 
# # make display
# display = FeatureAgglomerationDisplay(Ximg, pipe)
# plist = []
# for step_num in range(3):
#     plist.append(display.plot_dendrogram(step_num))
#     plist.append(display.plot_in_corr_map(step_num).fig)
# plist.append(display.plot_out_corr_map().fig)
# 
# # set sizes
# # for p, size in zip(plist, [(5, 3), (8, 8), (5, 5)]):
# #     p.set_size_inches(size)
# # plt.show()
# # plt.close()
# 
# # save to pdf
# save_multi_page_pdf(plist, "iterative_w_pipeline.pdf")
# 
# # %% [markdown]
# # ## Test IterativeCorrFeatureAgglomeration w/o Pipeline
# 
# # %%
# # test for IterativeCorrFeatureAgglomeration
# Ximg = ScaledVarianceThreshold().fit_transform(SimpleImputer().fit_transform(X.filter(like="image8_")))
# pipe = IterativeCorrFeatureAgglomeration(r=0.8)
# pipe.fit(Ximg)
# 
# # make display
# display = FeatureAgglomerationDisplay(Ximg, pipe)
# plist = []
# for step_num in range(3):
#     plist.append(display.plot_dendrogram(step_num))
#     plist.append(display.plot_in_corr_map(step_num).fig)
# plist.append(display.plot_out_corr_map())
# 
# # set sizes
# # for p, size in zip(plist, [(5, 3), (8, 8), (5, 5)]):
# #     p.set_size_inches(size)
# # plt.show()
# # plt.close()
# 
# # save to pdf
# save_multi_page_pdf(plist, "iter_no_pipeline.pdf")
