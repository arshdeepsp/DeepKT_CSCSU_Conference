import os
import numpy as np

class DataAssistMatrix:
    def __init__(self, params=None):
        print("Loading khan...")

        # Optionally set max_steps and max_train from params if provided.
        self.max_steps = params.get('max_steps') if params and 'max_steps' in params else None
        self.max_train = params.get('max_train') if params and 'max_train' in params else None

        # For the training phase.
        root = os.path.join('..', 'data', 'assistments')
        train_path = os.path.join(root, 'builder_train.csv')
        train_data = []
        longest = 0
        total_answers = 0

        # Initialize questions as a set to track unique question ids.
        self.questions = set()
        self.n_questions = 0
        
        with open(train_path, "r") as file:
            while True:
                student = self.load_student(file)
                if student is None:
                    break
                if student['n_answers'] >= 2:
                    train_data.append(student)
                if len(train_data) % 100 == 0:
                    print(len(train_data))
                if student['n_answers'] > longest:
                    longest = student['n_answers']
                total_answers += student['n_answers']
        self.trainData = train_data
        

        # For the test phase.
        # (Note: The Lua code reinitializes self.questions here.)
        # self.questions = set()
        # self.n_questions = 0
        test_path = os.path.join(root, 'builder_test.csv')
        test_data = []
        with open(test_path, "r") as file:
            while True:
                student = self.load_student(file)
                if student is None:
                    break
                if student['n_answers'] >= 2:
                    test_data.append(student)
                if len(test_data) % 100 == 0:
                    print(len(test_data))
                # In the Lua code, the longest update here is commented out.
                total_answers += student['n_answers']
        self.testData = test_data

        print("total answers", total_answers)
        print("longest", longest)
        print("total test questions: ", self.questions)
        print("total test questions number: ", len(self.questions))
        print("total train questions number: ", self.n_questions)
        
        # Hongmin start
        # After processing training data:
        # sorted_qids_train = sorted(list(self.questions))
        # self.question_mapping_train = {qid + 1: i + 1 for i, qid in enumerate(sorted_qids_train)}
        # self.n_questions_train = len(sorted_qids_train)
        sorted_qids = sorted(list(self.questions))
        self.question_mapping = {qid + 1: i + 1 for i, qid in enumerate(sorted_qids)}
        self.n_questions = len(sorted_qids)


        # Hongmin end

    def load_student(self, file):
        """
        Reads three lines from the file:
          1. The number of answers (nStepsStr)
          2. A comma-separated list of question IDs
          3. A comma-separated list of correctness values
        Returns a dictionary representing the student or None if EOF is reached.
        """
        n_steps_str = file.readline()
        question_id_str = file.readline()
        correct_str = file.readline()

        # If any of the three lines is empty, we've reached EOF.
        if not n_steps_str or not question_id_str or not correct_str:
            return None

        n_steps_str = n_steps_str.strip()
        question_id_str = question_id_str.strip()
        correct_str = correct_str.strip()

        try:
            n = int(n_steps_str)
        except ValueError:
            return None

        # Override n with max_steps if specified.
        if self.max_steps is not None:
            n = self.max_steps

        student = {}

        # Create an array for question IDs. (Lua adds 1 to each id.)
        student['questionId'] = np.zeros(n, dtype=np.uint8)
        q_ids = question_id_str.split(",")
        for i, id_str in enumerate(q_ids):
            if i >= n:
                break
            try:
                qid = int(id_str)
            except ValueError:
                qid = 0
            student['questionId'][i] = qid + 1  # Lua indexing adjustment

            # Update questions set and count if this question id hasn't been seen.
            if qid not in self.questions:
                self.questions.add(qid)
                self.n_questions += 1

        # Create an array for correctness values.
        student['correct'] = np.zeros(n, dtype=np.uint8)
        correct_vals = correct_str.split(",")
        for i, val in enumerate(correct_vals):
            if i >= n:
                break
            try:
                student['correct'][i] = int(val)
            except ValueError:
                student['correct'][i] = 0

        student['n_answers'] = n
        return student

    def getTestData(self):
        return self.testData

    def getTrainData(self):
        return self.trainData

    def getTestBatch(self):
        # Return a shallow copy of testData.
        return list(self.testData)
