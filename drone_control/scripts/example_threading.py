import threading
import time
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class MyClass:
    def __init__(self):
        self.val = 0

    @threaded
    def func_to_be_threaded(self):
        for i in range(10):
            self.val += 1
            print("Thread: %d" % self.val)
            time.sleep(1)


def main():
    myObj = MyClass()
    print(myObj.val)

    # one call to function
    handle = myObj.func_to_be_threaded()
    for i in range(10):
        print("Main: %d" % myObj.val)
        time.sleep(0.25)
    handle.join()

if __name__ == '__main__':
    main()