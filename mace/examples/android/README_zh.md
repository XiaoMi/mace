# mace demo使用方法

* 使用前需要生成静态库和头文件,具体参考[文档](docs)
    * 把mace/public目录下的mace.h和mace_runtime.h拷贝到macelibrary/src/main/cpp/include下面
    * 把生成的mace/codegen/engine/mace_engine_factory.h拷贝到macelibrary/src/main/cpp/include下面
    * 静态库的路径是在mace/build/demo_app_models/lib/下
* 使用android studio 导入项目,然后运行install run
* 还可以使用gradle命令(需要安装gradle)生成apk 具体命令例如:./gradlew assemble(或者Release|Debug)

## 交流与反馈
* 欢迎通过Github Issues提交问题报告与建议
* QQ群: 756046893

## License
[Apache License 2.0](LICENSE).
