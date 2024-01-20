import re


class csv_reader:
    def __init__(self, file, read_line=False):
        """
        Initializes the reader, reading through any potential header,
        and saving the column which contains the ECG data either by the column labels
        or through prompting the user.
        :param file: The file which will be read from
        """
        if isinstance(file, str):
            file = open(file, 'r+')
        self.file = file
        self.read_header()
        self.read_whole_line = read_line
        loc = self.file.tell()
        temp_line = self.file.readline()
        self.file.seek(loc)
        temp_items = re.split(',\\s*', temp_line)
        self.columns = len(temp_items)
        self.column = -1
        for i in range(self.columns):
            if 'ECG' in temp_items[i] or 'ecg' in temp_items[i]:
                self.column = i
        if self.column == -1 and not read_line:
            self.ecg_column = self.prompt_user()
        else:
            self.file.readline()

    def read_line(self):
        """
        Reads a line and returns the ECG data.
        :return: The ECG from the line
        """
        line = self.file.readline()
        if self.read_whole_line:
            return line
        if len(line) == 0:
            raise StopIteration
        items = re.split(',\\s*', line)
        try:  # Handles the few cases where there might be multiple string labels for columns
            if 'x' in items[self.column]:
                return 0
            return float(items[self.column])
        except ValueError:
            return self.read_line()

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

        def prompt_helper():
            """
            Helper function that deals with misinputs
            :return: The column that the user selects
            """
            try:
                ecg_column = int(input('Which column contains ECG data (first column as column 1):  '))
            except ValueError:
                print("Please input a proper value")
                ecg_column = prompt_helper()
            if ecg_column <= 0 or ecg_column > self.columns:
                print("Please input a proper value")
                ecg_column = prompt_helper()
            return ecg_column

        self.file.seek(loc)
        return prompt_helper() - 1

    def __iter__(self):
        """
        Returns itself as the iterator object
        """
        return self

    def __next__(self):
        """
        Returns the next value from the file (for the purpose of an iterator)
        :return: The next value from the file (as defined by read_line), and raises an StopIteration at end of file.
        """
        return self.read_line()


if __name__ == '__main__':
    reader = csv_reader('../ECG_Data/T21_transition example3_900s.ascii')
    i = 0
    for line in reader:
        i += 1
        if i > 20:
            break
        print(line)

