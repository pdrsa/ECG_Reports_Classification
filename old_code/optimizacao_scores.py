import pandas as pd
import re

print("Lendo arquivo...")
fulldata = pd.read_csv("../data/resultados/sem_siglas/resultado_sem_siglas", sep=";")

print("Trabalhando...")
stringHeader = 'idExame,idLaudo,área_eletricamente_inativa,Bloqueio_de_ramo_direito,Bloqueio_de_ramo_esquerdo,Bloqueio_de_ramo_direito_e_bloqueio_divisional_anterossuperior_do_ramo_esquerdo,Bloqueio_intraventricular_inespecífico,Sobrecarga_ventricular_esquerda_(critérios_de_Romhilt-Estes),Sobrecarga_ventricular_esquerda_(critérios_de_voltagem),Fibrilação_atrial,Flutter_atrial,Bloqueio_atrioventricular_de_2°_grau_Mobitz_I,Bloqueio_atrioventricular_de_2°_grau_Mobitz_II,Bloqueio_atrioventricular_2:1,Bloqueio_atrioventricular_avançado,Bloqueio_atrioventricular_total,Pré-excitação_ventricular_tipo_Wolff-Parkinson-White,Sistema_de_estimulação_cardíaca_normofuncionante,Sistema_de_estimulação_cardíaca_com_disfunção,Taquicardia_atrial_multifocal,Taquicardia_atrial,Taquicardia_supraventricular,Corrente_de_lesão_subendocárdica,Alterações_primárias_da_repolarização_ventricular,Extrassístoles_supraventriculares,Extrassístoles_ventriculares,Bradicardia_sinusal,ECG_dentro_dos_limites_da_normalidade_para_idade_e_sexo,Alterações_da_repolarização_ventricular_atribuídas_à_ação_digitálica,Alterações_inespecíficas_da_repolarização_ventricular,Alterações_secundárias_da_repolarização_ventricular,Arritmia_sinusal,Ausência_de_sinal_eletrocardiográfico_que_impede_a_análise,Interferência_na_linha_de_base_que_não_impede_a_análise_do_ECG,Ausência_de_sinal_eletrocardiográfico_que_não_impede_a_análise,Traçado_com_qualidade_técnica_insuficiente,Possível_inversão_de_posicionamento_de_eletrodos,Baixa_voltagem_em_derivações_precordiais,Baixa_voltagem_em_derivações_periféricas,Bloqueio_atrioventricular_de_1°_grau,Bloqueio_de_ramo_direito_e_bloqueio_divisional_posteroinferior_do_ramo_esquerdo,Bloqueio_divisional_anterossuperior_do_ramo_esquerdo,Bloqueio_divisional_posteroinferior_do_ramo_esquerdo,Desvio_do_eixo_do_QRS_para_direita,Desvio_do_eixo_do_QRS_para_esquerda,Dissociação_atrioventricular_isorrítmica,Distúrbio_de_condução_do_ramo_direito,Distúrbio_de_condução_do_ramo_esquerdo,Intervalo_PR_curto,Intervalo_QT_prolongado,Isquemia_subendocárdica,Progressão_lenta_de_R_nas_derivações_precordiais,Pausa_sinusal,Corrente_de_lesão_subepicárdica,Corrente_de_lesão_subepicárdica_-_provável_infarto_agudo_do_miocárdio_com_supradesnivelamento_de_ST,Repolarização_precoce,Ritmo_atrial_ectópico,Ritmo_atrial_multifocal,Ritmo_idioventricular_acelerado,Ritmo_juncional,Síndrome_de_Brugada,Sobrecarga_atrial_direita,Sobrecarga_atrial_esquerda,Sobrecarga_biatrial,Sobrecarga_biventricular,Sobrecarga_ventricular_direita,Sobrecarga_ventricular_esquerda(_critérios_de_voltagem),Taquicardia_sinusal,Taquicardia_ventricular_não_sustentada,Taquicardia_ventricular_sustentada,Suspeita_de_Síndrome_de_Brugada_repetir_V1-V2_em_derivações_superiores,Taquicardia_juncional,Batimento_de_escape_atrial,Batimento_de_escape_supraventricular,Batimento_de_escape_juncional,Batimento_de_escape_ventricular'

scores = [re.sub('\[|\]','',x) for x in fulldata[" resultado"]]

idLaudos   = fulldata["idLaudo"]
idExames   = fulldata["idExame"]

idExames = idExames.tolist()
idLaudos = idLaudos.tolist()

arquivo = open('../data/resultados/sem_siglas/resultado_sem_siglas_scores.csv', 'w')
arquivo.write(stringHeader)
arquivo.write("\n")
print("Escrevendo no arquivo...")
for i in range(len(scores)):
    arquivo.write(str(idExames[i]))
    arquivo.write(',')
    arquivo.write(str(idLaudos[i]))
    arquivo.write(',')
    arquivo.write(scores[i])
    arquivo.write("\n")
arquivo.close()
