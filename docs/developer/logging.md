Logging
=======

The rule of VLOG:
0. Ad hoc debug logging, should only be added in test or temporary ad hoc debugging
1. Important network level Debug/Latency trace log (Op run should never generate level 1 vlog)
2. Important op level Latency trace log
3. Unimportant Debug/Latency trace log
4. Verbose Debug/Latency trace log
