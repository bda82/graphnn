from dataclasses import dataclass


@dataclass
class Urls:
    citation_url = "https://github.com/tkipf/gcn/raw/master/gcn/data/{}"


def urls_fabric():
    return Urls()
