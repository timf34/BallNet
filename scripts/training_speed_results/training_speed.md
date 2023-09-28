## Notes from 26/11/22 

_First implementation of benchmark and mixed precision (AMP)_

Results with:
- Batch size = 1
- Num workers = 2

1st run:
```bash 
Default precision training:
Total execution time = 53.797 sec
Max memory used by tensors = 369454080 bytes

Mixed precision training:
Total execution time = 58.560 sec
Max memory used by tensors = 518700544 bytes
```

2nd run:
```bash 
Default precision training:
Total execution time = 66.260 sec
Max memory used by tensors = 518700544 bytes

Mixed precision training:
Total execution time = 60.666 sec
Max memory used by tensors = 369454080 bytes
```

3rd run, with the order or execution swapped -> mixed precision first, then default precision

```bash 
Mixed precision training:
Total execution time = 37.770 sec
Max memory used by tensors = 369454080 bytes 

Default precision training:
Total execution time = 38.634 sec
Max memory used by tensors = 518700544 bytes 
```

4th run, same order as above (mixed first, then default):

```bash 
Mixed precision training:
Total execution time = 37.804 sec
Max memory used by tensors = 369454080 bytes 

Default precision training:
Total execution time = 40.336 sec
Max memory used by tensors = 518700544 bytes
```

Changing the config to `batch_size=2` and `num_workers=2`:

```bash

Mixed precision training:
Total execution time = 51.359 sec
Max memory used by tensors = 598464512 bytes 

Default precision training:
Total execution time = 53.005 sec
Max memory used by tensors = 913709568 bytes 
```

Changing the config to `batch_size=2` and `num_workers=2`:

```bash

Mixed precision training:
Total execution time = 64.767 sec
Max memory used by tensors = 598464512 bytes 

Default precision training:
Total execution time = 66.259 sec
Max memory used by tensors = 913709568 bytes 
```

Note: I'm not sure what accounts for the variance here between runs... but what we can see is that the mixed precision is always faster
and using less memory 

Note: this setup didn't work when I set num_wokers to 8, and batch_size to 8... but that's probably just due to my computer. 
I need to test this on an AWS instance and see the results there!

Note: Although the speed is dropping when I up the batch_size, do note that we are getting _n_x the data throughput 
as I am breaking based on number of epochs covered, not images/ data covered. 