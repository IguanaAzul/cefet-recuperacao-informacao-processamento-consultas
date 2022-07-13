from datetime import datetime


class CheckTime(object):
    def __init__(self):
        self.time = datetime.now()

    def finish_time(self):
        delta = datetime.now() - self.time
        self.time = datetime.now()
        return delta

    def print_delta(self, task):
        delta = self.finish_time()
        print(task + " done in " + str(delta.total_seconds()))
