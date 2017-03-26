import bisect

class ValueWrapper:
    def __init__(self, value):
        self.value = float(value)

    def __lt__(self, other):
        return self.value < float(other[0])

    def __gt__(self, other):
        return self.value > float(other[0])

    def __eq__(self, other):
        return self.value == float(other[0])

class BiSearch:
    @staticmethod
    def search(a, b):
        return bisect.bisect_right(a, ValueWrapper(b))

if __name__ == '__main__':
    a = [['12345', 0, 0], ['12346', 0, 1], ['12348', 0, 2]]
    b = '12346'
    idx = BiSearch.search(a, b)
    print 'idx:', idx, 'value:', a[idx]
    
