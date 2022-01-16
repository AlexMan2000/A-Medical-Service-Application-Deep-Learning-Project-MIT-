
from question_classifier import *
from question_parser import *
from answer_search import *

'''QA'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = "Hello, I am your Medical Assistant, I hope I can help you. If I don't answer it, I suggest you consult a professional doctor. Wish a great body!"
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)

if __name__ == '__main__':
    handler = ChatBotGraph()

    "在这里和Tinkter连接"
    while 1:

        '''从前端获取用户输入
        记得好像是button.text()什么的,传到后端'''
        question = input('User:')

        "后端以字符串形式接受question, 获取answer返回前端"
        answer = handler.chat_main(question)
        print('Medic:', answer)

