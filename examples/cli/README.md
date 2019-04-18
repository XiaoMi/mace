Examples
=======

* Convert model

```
python tools/converter.py convert --config=/path/to/your/model_deployment_file
```

* Run example
```
python tools/converter.py run --config=/path/to/your/model_deployment_file --example
```

* Validate result
```
python tools/converter.py run --config=/path/to/your/model_deployment_file --example --validate
```

* Check the logs
```
adb logcat
```
