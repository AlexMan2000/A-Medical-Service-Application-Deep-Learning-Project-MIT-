
import os
import json
import pandas as pd
import numpy as np
from py2neo import Graph,Node,Relationship


class MedicalKnowledgeGraph:
    def __init__(self):

        # connect to neo4j
        self.graph = Graph('http://localhost:7474',auth=("neo4j","123456"))
        self.dataPath = r"C:\Users\DELL\Desktop\MIT_Final_Project\Core\data\knowledgeGraph\medical_knowledge_base.csv"


    def read_nodes(self):
        # three types of nodes
        departments = []
        diseases = []
        symptoms = []

        disease_infos = []

        # Building relationship between nodes
        rels_department = []  # relationship between departments

        rels_symptom = []  # relationship between symptons

        rels_acompany = []  # relationship between diseases and their accompany

        rels_category = []  # relationship between disease and department


        df = pd.read_csv(self.dataPath, encoding='gbk')

        count = 0

        # Iterate through every line.
        for index, row in df.iterrows():
            disease_dict = {}
            count += 1
            print(count)
            disease = row['name']
            disease_dict['name'] = disease
            diseases.append(disease)

            disease_dict['desc'] = ''

            disease_dict['prevent'] = ''

            disease_dict['cause'] = ''

            disease_dict['easy_get'] = ''

            disease_dict['cure_department'] = ''

            disease_dict['cure_way'] = ''

            disease_dict['cure_lasttime'] = ''


            disease_dict['symptom'] = ''


            disease_dict['cured_prob'] = ''

            symtom_temp = row['symptom'].replace('[', '').replace(']', '').replace("'", '').split(",")
            symptoms += symtom_temp
            for symptom in symtom_temp:
                rels_symptom.append([disease, symptom])

            acompany_temp = row['acompany'].replace('[', '').replace(']', '').replace("'", '').split(",")
            for acompany in acompany_temp:
                rels_acompany.append([disease, acompany])

            disease_dict['desc'] = row['desc']

            disease_dict['prevent'] = row['prevent']

            disease_dict['cause'] = row['cause']

            # disease_dict['get_prob'] = row['get_prob']
            #
            # disease_dict['easy_get'] = row['easy_get']

            cure_department = row['cure_department'].replace('[', '').replace(']', '').replace("'", '').split(",")
            if len(cure_department) == 1:
                rels_category.append([disease, cure_department[0]])
            if len(cure_department) == 2:
                big = cure_department[0]
                small = cure_department[1]
                rels_department.append([small, big])
                rels_category.append([disease, small])

            disease_dict['cure_department'] = cure_department
            departments += cure_department

            disease_dict['cure_way'] = row['cure_way'].replace('[', '').replace(']', '').replace("'", '').split(",")

            # disease_dict['cure_lasttime'] = row['cure_lasttime']
            #
            # disease_dict['cured_prob'] = row['cured_prob']

            disease_infos.append(disease_dict)
        return set(departments), set(symptoms), set(diseases), disease_infos, \
               rels_department, \
               rels_symptom, rels_acompany, rels_category

    '''建立节点'''

    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.graph.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点'''

    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            if disease_dict["cause"] is np.nan:
                disease_dict["cause"] = ""

            if disease_dict["prevent"] is np.nan:
                disease_dict["prevent"] = ""
            node = Node("Disease", name=disease_dict['name'], desc=disease_dict['desc'],
                        prevent=disease_dict['prevent'], cause=disease_dict['cause'],
                        cure_department=disease_dict['cure_department']
                        , cure_way=disease_dict['cure_way'])
            self.graph.create(node)
            count += 1
            print(count)
        return

    '''创建知识图谱实体节点类型schema'''

    def create_graphnodes(self):
        Departments, Symptoms, Diseases, disease_infos, rels_department, rels_symptom, rels_acompany, rels_category = self.read_nodes()
        self.create_diseases_nodes(disease_infos)

        self.create_node('Department', Departments)
        print(len(Departments))

        self.create_node('Symptom', Symptoms)
        return

    '''创建实体关系边'''

    def create_graphrels(self):
        Departments, Symptoms, Diseases, disease_infos, rels_department, rels_symptom, rels_acompany, rels_category = self.read_nodes()

        self.create_relationship('Department', 'Department', rels_department, 'belongs_to', '属于')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Department', rels_category, 'belongs_to', '所属科室')

    '''创建实体关联边'''

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.graph.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''导出数据'''

    def export_data(self):

        Departments, Symptoms, Diseases, disease_infos, rels_department, rels_symptom, rels_acompany, rels_category = self.read_nodes()
        f_department = open('dict/department.txt', 'w+')
        f_symptom = open('dict/symptoms.txt', 'w+')
        f_disease = open('dict/disease.txt', 'w+')

        f_department.write('\n'.join(list(Departments)))
        f_symptom.write('\n'.join(list(Symptoms)))
        f_disease.write('\n'.join(list(Diseases)))

        f_department.close()
        f_symptom.close()
        f_disease.close()

        return

if __name__ == '__main__':
    handler = MedicalKnowledgeGraph()
    handler.create_graphnodes()
    handler.create_graphrels()
    # handler.export_data()