## Profiling guide
<!-- to do : add installs here  -->

To start a profiling With cProfile on the MNIST quick run example run : 

```bash
 declearn-split --folder "examples/mnist_quickrun" && python -m cProfile -o profiling/output.prof profiling/profile_run.py --config "examples/mnist_quickrun/config.toml"
```

The results will be saved in the profiling/output.prof  
for and interactive interpretation use pstats

```bash
python -m pstats profiling/output.prof
```

example usage 
 ```bash
profiling/output.prof% sort cumtime 
profiling/output.prof% stats 20
Tue Apr 14 10:59:27 2026    profiling/output.prof

         98419571 function calls (96772375 primitive calls) in 191.153 seconds

   Ordered by: cumulative time
   List reduced from 17664 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   4627/1    0.078    0.000  191.185  191.185 {built-in method builtins.exec}
        1    0.000    0.000  191.185  191.185 profiling/profile_run.py:1(<module>)
        1    0.000    0.000  189.182  189.182 /home/fslimani/work/declearn/declearn/quickrun/_run.py:292(main)
        1    0.000    0.000  189.182  189.182 /home/fslimani/.venvs/declearn311/lib/python3.11/site-packages/fire/core.py:73(Fire)
        1    0.000    0.000  189.181  189.181 /home/fslimani/.venvs/declearn311/lib/python3.11/site-packages/fire/core.py:361(_Fire)
        1    0.000    0.000  189.181  189.181 /home/fslimani/.venvs/declearn311/lib/python3.11/site-packages/fire/core.py:652(_CallAndUpdateTrace)
        1    0.000    0.000  189.179  189.179 /home/fslimani/work/declearn/declearn/quickrun/_run.py:284(fire_quickrun)
        1    0.000    0.000  189.179  189.179 /usr/lib/python3.11/asyncio/runners.py:158(run)
        1    0.000    0.000  189.179  189.179 /usr/lib/python3.11/asyncio/runners.py:86(run)
        3    0.000    0.000  189.178   63.059 /usr/lib/python3.11/asyncio/base_events.py:614(run_until_complete)
        3    0.001    0.000  189.178   63.059 /usr/lib/python3.11/asyncio/base_events.py:591(run_forever)
      548    0.012    0.000  189.177    0.345 /usr/lib/python3.11/asyncio/base_events.py:1834(_run_once)
     2283    0.004    0.000  172.476    0.076 /usr/lib/python3.11/asyncio/events.py:78(_run)
     2283    0.008    0.000  172.471    0.076 {method 'run' of '_contextvars.Context' objects}
      153    0.000    0.000  166.992    1.091 /home/fslimani/work/declearn/declearn/quickrun/_run.py:123(run_client)
      153    0.001    0.000  166.928    1.091 /home/fslimani/work/declearn/declearn/main/_client.py:210(async_run)
      123    0.002    0.000  166.449    1.353 /home/fslimani/work/declearn/declearn/main/_client.py:228(handle_message)
       60    0.001    0.000  149.668    2.494 /home/fslimani/work/declearn/declearn/main/_client.py:521(training_round)
       30    0.000    0.000  148.476    4.949 /home/fslimani/work/declearn/declearn/training/_manager.py:156(training_round)
       30    0.001    0.000  148.473    4.949 /home/fslimani/work/declearn/declearn/training/_manager.py:187(_training_round)

```
Breakdown : 
ncalls — how many times this function was called  
tottime — time spent inside this function only   
percall (first one) — tottime / ncalls. Average time per call for the function itself.  
cumtime — total time spent in this function including everything it called.  
percall (second one) — cumtime / ncalls. Average total time per call including sub-calls.  

# There is a tool called snakeviz to open an interactive broswer page to visualise the data 
usage : 
```bash
 snakeviz profiling/output.prof 
```


# Profiling with Py-spy
To launch run command 
```bash
declearn-split --folder "examples/mnist_quickrun" && py-spy record --subprocesses -o profiling/flamegraph.svg -- python profiling/profile_run.py --config "examples/mnist_quickrun/config.toml"
```  
The result is saved in the flamegraph.svg, preferably to open with a browser for an interactive interpretation. Py-spy captures the call stack regurarly so the wide bars at top are the ones who spent of the time in the CPU regardless of what they were doing  

There are other formats to use  
The speedscope format that shows actual time 
```bash
py-spy record --subprocesses --format speedscope -o profiling/speedscope.json -- python profiling/profile_run.py --config "examples/mnist_quickrun/config.toml"
```
to preview open : https://www.speedscope.app/ and upload the resulst speedscope.json  
There is also the raw format (useful for parsing and doing direct comparaison )
 ```bash
py-spy record --subprocesses --format raw -o profiling/speedscope.json -- python profiling/profile_run.py --config "examples/mnist_quickrun/config.toml"
```
in this format every line is a full call stack  

py-spy might be noisy because it interrupts for sampling in intervales that are not synchronised with the functions of the code so there might be a little bit of variance..either run multiple times or set a threshold to call something a regression 


# Profiling with Scalene

Command  
```bash 
declearn-split --folder "../examples/mnist_quickrun" && scalene --outfile scalene_report.html --html --cpu --memory --profile-all --- python scalene_run.py
```

does not work well for the quickexample directly its needs isolated benchmarking 


# Profiling with Line Profiler
not useful for now because among its limitations is asynchronus programming and multi-prcoessing, on top that to work the code needs to be decorated with @Profile which is impractical 

they suggested Yappi which is multithreading and asyncio aware 


# Pofiling with yappi 
it gives more precise insights because if it being multi-threading aware 

```bash 
declearn-split --folder "../examples/mnist_quickrun" && python yappi_run.py
```


## MEMORY PROFILING 
 
 # Memray 
  advantage is that its at C level so it tracks memory allocation and shows the python call stack so we know who is occupying the memory 

usage 
```bash
declearn-split --folder "../examples/mnist_quickrun" && memray run -o memray_output.bin  memray_run.py
```
Intrepreting the outputs  

flamegraph:
```bash
memray flamegraph memray_output.bin -o memray_flamegraph.html
```
summary (table view): 
```bash
memray summary memray_output.bin
```
stats (rankings)
```bash
 memray stats memray_output.bin
```



