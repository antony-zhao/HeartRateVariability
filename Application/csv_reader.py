import re


class csvReader:
    def __init__(self, file):
        """
        Initializes the reader, reading through any potential header,
        and saving the column which contains the ECG data either by the column labels
        or through prompting the user.
        :param file: The file which will be read from
        """
        self.file = file
        self.read_header()
        loc = self.file.tell()
        temp_line = self.file.readline()
        self.file.seek(loc)
        temp_items = re.split(',\\s*', temp_line)
        self.columns = len(temp_items)
        self.column = -1
        for i in range(self.columns):
            if 'ECG' in temp_items[i] or 'ecg' in temp_items[i]:
                self.column = i
        if self.column == -1:
            self.ecg_column = self.prompt_user()
        else:
            self.file.readline()

    def read_line(self):
        """
        Reads a line and returns the ECG data.
        :return: The ECG from the line
        """
        line = self.file.readline()
        items = re.split(',\\s*', line)
        return float(items[self.column])

    def read_header(self):
        """
        Reads through the header by checking if the lines match the number of columns
        split by commas, and leaves the file there.
        """
        while True:
            loc = self.file.tell()
            consistent = True
            prev_column = None
            for _ in range(5):
                line = self.file.readline()
                columns = len(re.split(',\\s*', line))
                if ',' not in line:
                    consistent = False
                    break
                if prev_column is None or columns == prev_column:
                    prev_column = columns
                else:
                    consistent = False
                    break
            self.file.seek(loc)
            if consistent:
                return
            else:
                self.file.readline()

    def prompt_user(self):
        """
        Prints a few lines of data and prompts the user for which column contains the ECG data.
        :return: The index of the data (index 0 whereas prompt is index 1)
        """
        loc = self.file.tell()
        lines = []
        for _ in range(10):
            lines.append(self.file.readline())
        print('Data:')
        for line in lines:
            print(line, end='')
        ecg_column = int(input('Which column contains ECG data (first column as column 1):  '))
        self.file.seek(loc)
        return ecg_column - 1


if __name__ == '__main__':
    reader = csvReader(open('../ECG_Data/T21_transition example3_900s.ascii', 'r'))
    for i in range(20):
        print(reader.read_line())
