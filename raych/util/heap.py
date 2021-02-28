from typing import TypeVar
from typing_extensions import Protocol


C = TypeVar("C", bound="Comparable")


class Comparable(Protocol):
    "for fun, useless in this situation"

    def __eq__(self, other: C) -> bool:
        ...

    def __lt__(self: C, other: C) -> bool:
        ...

    def __gt__(self: C, other: C) -> bool:
        return (not self < other) and self != other

    def __le__(self: C, other: C) -> bool:
        return self < other or self == other

    def __ge__(self: C, other: C) -> bool:
        return not self < other


class CheckpointStruct(Comparable):
    def __init__(self, score, name):
        self.score = score
        self.name = name

    def __gt__(self, other):
        return (not self.score < other.score) and self.score != other.score

    def __le__(self: C, other: C) -> bool:
        return self.score < other.score or self.score == other.score

    def __ge__(self: C, other: C) -> bool:
        return not self.score < other.score

    def __hash__(self) -> int:
        return hash((self.score, self.name))

    def __repr__(self) -> str:
        # # useful hash func
        # for c in some_string:
        #   hash = 101 * hash  +  ord(c)
        return str(self.__class__.__name__ + "::" + str(self.score) + "::" + self.name)


class Heap:
    def __init__(self, items=[]):
        self.n = 0
        self.heap = [None]  # heap[0]没有实际意义，只为方便index计算
        self.rank = {}      # 查找堆中元素的下标index
        for x in items:
            self.push(x)

    def __len__(self):
        return len(self.heap) - 1

    def push(self, x):
        assert x not in self.rank
        end = len(self.heap)
        self.heap.append(x)  # 尾部添加并上浮，保持堆排序
        self.rank[x] = end
        self.up(end)

    def pop(self):
        root = self.heap[1]
        del self.rank[root]
        x = self.heap.pop()  # 尾部pop元素，重新从堆顶下沉，保持堆排序
        if len(self.heap) != 0:  # 堆非空
            self.heap[1] = x
            self.rank[x] = 1
            self.down(1)
        return root

    def up(self, i):
        x = self.heap[i]
        while i > 1 and x < self.heap[i//2]:
            self.heap[i] = self.heap[i//2]
            self.rank[self.heap[i//2]] = i
            i //= 2
        self.heap[i] = x
        self.rank[x] = i

    def down(self, i):
        x = self.heap[i]
        curLen = len(self.heap)
        while True:
            left = 2 * i
            right = left + 1
            if right < curLen and self.heap[right] < x and \
                    self.heap[right] < self.heap[left]:
                self.heap[i] = self.heap[right]
                self.rank[self.heap[right]] = i
                i = right
            elif left < curLen and self.heap[left] < x:
                self.heap[i] = self.heap[left]
                self.rank[self.heap[left]] = i
                i = left
            else:
                self.heap[i] = x
                self.rank[x] = i
                return

    def update(self, old, new):
        "更改元素old为new"
        i = self.rank[old]
        del self.rank[old]
        self.heap[i] = new
        self.rank[new] = i
        if old < new:
            self.down(i)
        else:
            self.up(i)


if __name__ == "__main__":
    hh = Heap([CheckpointStruct(1, "as")])
    hh.push(CheckpointStruct(2, "bb"))
    hh.push(CheckpointStruct(3, "cc"))
    hh.push(CheckpointStruct(4, "dd"))

    for it in hh.heap[1:]:
        print(it)

    top = hh.pop()
    print(top)
