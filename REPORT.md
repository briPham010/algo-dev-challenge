# Overview
This project implements and evaluates an RSI-based trading strategy adapted from TradingView’s Pine Script V5 adjusted for an event-driven Python back testing environment known as at Nautilus Trader. The primary goals of this project were to:

- Accurately adapt the strategy into Python's Nautilus Trader
- Optimize key parameters and interpret results in light of turnover and execution assumptions without obvious overfitting
- Select a final strategy based on realism and robustness

# Implementation
Implementing the original strategy from Pine Script came with nothing but challenges. I  placed a significant emphasis on “understanding” each step of the strategy before I attempted any sort of implementation. Initially, I implemented each feature (entry/exit logic, pyramiding logic etc.) following the provided scaffold and Pine Script very closely, but this created a plethora of challenges and bugs and as well a unique journey for finding and fixing said bugs. 

## Execution Timing
As someone with a basic understanding of PineScript (much more now after this project), I was informed not-so-early that Pine Script evaluates strategy logic as if state changes occur within the bar, meaning repeated calls to strategy.entry are idempotent. However, in Nautilus Trader’s event driven engine, I found out following the logic to a tee results in an abundance of orders and actions happening within the same bar, causing nothing short of chaos. My solution to this was to enforce a very strict “one action per bar” policy, ensuring that after a successful action (e.g. entry, exit, pyramid), no additional logic was to run on the same bar as we proceed to the next bar.

## Temporal Leakage
As surprising as it sounds, I thought RSI divergence was one of the easier features to implement in this project but that still didn’t make the process any much simpler. A majority of the time spent implementing involved trying to understand what “higher highs”, “lower lows” and etc. actually meant in addition to understanding Pine Script’s pivothigh and pivotlow functions. I opted to recreate the functions in python for simplicity. This proved to be very beneficial since implementation was seamless after I fully grasped the idea of RSI divergence. Although it sounded seamless, implementing RSI divergence did not come without its challenges, many of my bugs were traced back to the “rolling window” I used which introduced temporal leakage and as well as naively treating bullish divergence as a state across multiple bars. A session of bug hunting, pages of log statements, and many sanity checks later would help me track these down.

## Pyramid Logic and Position/Sizing Errors
If I were to talk about all the setbacks experienced throughout this project, this report would span pages, but I’ll discuss a little bit about my implementation of the pyramiding logic since this feature single-handedly pushed my critical thinking and as well as attention-to-detail to the absolute maximum. This implementation started with understanding, a recurring pattern that I found helpful throughout this project, following the Pine Script and scaffold faithfully helped build a solid framework, but it was a lesson learned that Pine Script logic behaves much differently than the event driven system I was working with. This led to a lot of trigger issues such as with pyramids too frequently, position/leverage sizing issues, and position states not being updated in certain places. Initially, these bugs ran undetected until I sought some insight from the Square Kettle Team suggesting some factors on why my initial backtests were producing unrealistic results. Such factors led me back to the pyramid logic which unveiled the bugs hiding in broad daylight. Again, utilizing PineScript and Nautilus Trader’s documentation, stacks of log statements, sanity checks, and enough coffee to faint a small animal, the pyramid logic was restructured resulting in the model producing more realistic results.

## Optimization
Earlier, I mentioned that the model was producing unrealistic results, but this was not before I explored optimization. Understanding optimization wasn’t something I focused on due to my background in statistics and the fact that this isn’t the first model I’ve built and optimized (and hopefully not the last either), but understanding how each parameter affects the ratios produced definitely helped speed up the process. For starters, after playing around with a couple of optimizing techniques, I opted for Bayesian Optimization, for a variety of factors. I knew I wanted to optimize a pretty high-dimensional set of parameters, and I noticed that the back test was already computationally expensive. Unlike Grid Search which probably would still not finish by the time this report is done or Random Search that relies on luck, I determined that Bayesian optimization would be able to track down the global maximum. For the parameter set, I wanted to optimize RSI, long entry, long exit, atr period, sensitivity, and the base pyramid multiplier value. The initial three are standard, they make up the bread and butter of the strategy and greatly affect how the model performs. I chose to optimize atr period since it measures volatility, something I noticed was very abundant in the data, I wanted the data to tell me if the instruments required more smooth volatility over a noisy one to help avoid false positives. I included sensitivity and the pyramid base multiplier since I wanted to see if the optimizer could find an efficient sizing model while also accounting for recovery and draw-down risk. For the objective function, I opted for Sharpe Ratio since it prioritizes the stability and consistency of returns over the entire testing period, this helped keep my goal basic and obtainable. However, in light of risk. I noticed that in some back testing results, the model would produce some catastrophic losses ($40k+). An idea I had was to create an objective function that consists of weighted Sharpe and Calmar scores, but without the time, I opted for the safer Sharpe Ratio. f that, some research also introduced me to “Calmar’s Ratio”, although it wasn’t my main objective function, it was something I monitored since it monitors maximum draw-down indicating. Finally, I opted to do a walk forward optimization, utilizing a 67/33 split from January 2022 to December 2023 as the training split, and January 2024 to Deccember 2024 as the testing split. This prevents the model from overfitting and helps determine if the strategy is truly robust. The following are the final parameter ranges and settings used for the optimization:
```
RSI - [7, 21]
Long Entry - [20.0, 45.0]
Long Exit - [60.0, 80.0]
ATR Period - [10, 20]
Sensitivity Base - [0.2, 1.5]
Pyramid Multiplier Base - [1.0, 2.5]

res = gp_minimize(
    func=backtest,
    dimensions=param_ranges,
    n_calls=50,
    n_random_starts=10,
    random_state=6767,
)
```

## Results and Interpretation
Initially after running Bayesian Optimization, I observed a consistent pattern, many parameter combinations would produce unusually strong metrics (eg. very high Sharpe, Sortino, Calmar ratios of 20+ each) while also producing massive trade counts (40k+). Initially, I thought nothing of it since I knew that commissions weren’t being taken into account, but some insight from the Square Kettle Team suggested that something was massively wrong. This led to the refactoring of the original algorithm and the discovery of many other bugs. Nonetheless, even after the bug fixes, the same pattern kept arising, this led me to ask: should the strategy be treated as a scalper (high-frequency) or a more constrained mean reversion strategy (low-frequency) as originally intended by the Pine Script? Instead of forcing a single answer, I followed the guidance by The Square Kettle Team to investigate results for both approaches. 

## Optimization Results
Bayesian optimization results (see figures 1-3) consistently converges to a narrow region of the parameter space centered around 10-12 RSI with a long entry threshold in the low 30s, and a long exit threshold approaching 60. I chose to visualize the behavior on a stability graph (see figure 2) as it would also give me an indicator in the event of parameter overfitting. Using Sharpe ratio as the objective function, the best configuration reported was:

```
- RSI Period: 10
- Long Entry: 31.3
- Long Exit: 60
- ATR Period: 18
- Sensitivity: 0.2
- Pyramid Multiplier: 2.5

```
the results included headline metrics such an extremely high Sharpe and Calmar Ratios, alongside a massive number of trades. While these metrics are very impressive and even more so that it performed positively out-of-sample, it suggests that these parameters behave more as a high-frequency or a scalping interpretation of the strategy.

![Figure 1](/images/results1.png)

*Figure 1: Top 10 Bayesian optimization configurations ranked by Sharpe ratio along with other statistical metrics.*


![Figure 2](/images/stabilitychart1.png)

*Figure 2: Optimization stability surface for RSI Period vs. Long Entry threshold. Higher values indicate stronger backtest performance.*

![Figure 3](/images/out_of_sample_test.png)

*Figure 3: Backtest result using the optimal configuration denoted by the Bayesian optimizer.*


Additionally, as an experiment, I wanted to see if the optimizer would behave differently if pyramiding was disabled (see figures 3.1-3.2). Surprisingly, the optimizer also attempts to push the ranges (especially RSI) towards a single value like its pyramid enabled counterpart.

![Figure 3.1](/images/pyramid_disabled.png)

*Figure 3.1: Top 10 Bayesian optimization configurations tuned without pyramid logic ranked by Sharpe ratio along with other statistical metrics.*

![Figure 3.2](/images/pyramid_disabled_stability.png)

*Figure 3.2: Optimization stability surface for RSI Period vs. Long Entry threshold. Results from configuration tuned without pyramid logic.*

## Summary and Back Test Results
Due to computational setbacks, I chose to run the final backtests from January 2022 to December 2024. Including 2022 was very important here since it contained the 2022 stock market decline, if the model could bounce back from such a decline, it would be more likely to be successful in future tests.

### A). Scalping (Optimized Parameters)
In the high-frequency scalping style denoted by figures 1 and 2, the strategy triggers frequently and bullish divergence appears to act as a more permissive catalyst which results in pyramiding logic firing more frequently. The main benefit of this strategy is that it makes very consistent positive returns, even in out-of-sample results, the strategy holds and produces results in the testing environment. However, I anticipate that it would be very sensitive to execution and commission factors. The large number of trades would certainly reduce the effectiveness of such a strategy.


![Figure 4](/images/backtest_results_1.png)

*Figure 4: Full backtest run on optimized parameters as chosen by the Bayesian optimizer. (Scalping Model)*

### B). Constrained/Non-Scalping (Default Parameters)
In the constrained interpretation, the strategy is based on the intended logic from the original Pine Script. It enforces a much more realistic behavior by avoiding rapid-firing trades and pyramids, producing results that are very interpretable; indicating clear casual links between signal conditions and trades. As note, I've included the default parameters used below:

```
- RSI Period: 14
- Long Entry: 31
- Long Exit: 83
- ATR Period: 14
- Sensitivity: 0.3
- Pyramid Multiplier: 2
```

![Figure 5](/images/backtest_results_2.png)

*Figure 5: Full backtest run with default parameters. (Constrained Model)*

In short, I’ve opted for the constrained / non-scalping strategy on the default parameters since it satisfies the criteria that matters the most in a realistic scenario. For starters, it’s a very faithful translation of the original Pine Script. But it produces a more plausible trade frequency and risk statistics and is less dependent on execution assumptions (eg. commission fees). The pyramid behavior is much more predictable and respectful, reducing unwanted risk. This strategy is much more likely to remain robust under further out-of-sample testing than its high frequency counterpart.

# Conclusion
This project proved that simply translating a strategy is not a mechanical coding task, it’s disguised as a highly critical engineering and reasoning problem. The majority of time I spent understanding and debugging resulted in much more growth than what any typical leetcode or technical interview could ever offer. Overall, I believe this submission reflects not just the final strategy performance, but the process of building confidence in correctness, identifying failure modes, and making deliberate trade-offs, which is ultimately what I believe real algorithm development demands.


