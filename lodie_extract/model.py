class Record:

    def __init__(self, day):
        self.day = day
        self.items = []


class Item:

    def __init__(self, txt, users):
        self.txt = txt
        self.users = users


class Uids:

    def __init__(self):
        self.users = {"No Source":0}
        self.uid_counter = 1

    def getUid(self, username):
        if username in self.users:
            return self.users[username]
        else:
            self.users[username] = self.uid_counter
            self.uid_counter += 1
            return self.uid_counter - 1