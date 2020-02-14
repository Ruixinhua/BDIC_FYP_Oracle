# dataset

This dataset directory contains four sub directories:
- png: includes images with png or jpg format
- svg: includes svg format file
    - oracle inscriptions(甲骨文)
    - bronze inscription(青铜铭文即金文)
    - liu shu tong(六书通)
    - shuo wen jie zi(说文解字)
- [make_directory.py](make_directory.py): create png and svg directories with corresponding files.
- [statistics.py](statistics.py): after run make_directory.py, this script will give a statistics information of created png directory.

|          | categories | the total number |
| -------- | ---------- | ---------------- |
| 甲骨文   | 918        | 23855            |
| 金文     | 1505       | 20349            |
| 六书通   | 3548       | 24136            |
| 说文解字 | 5258       | 22698            |

