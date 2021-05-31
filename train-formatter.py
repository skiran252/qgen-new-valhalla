import json
import logging
import nltk
qg_format="highlight"
def _get_correct_alignement( context, answer):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()
    
def process_qa_text( context, question, answer):
    ans_gen_input = f"question: {question}  context: {context}"
    ans_gen_target = f"{answer}"
    return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

def process_qg_text( context, question, answer):
    answer_text = answer['text'].strip()
    
    if qg_format == "prepend":
        que_gen_input = f"answer: {answer_text}  context: {context}"
    elif qg_format == "highlight":
        start_pos, end_pos = _get_correct_alignement(context, answer)
        que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
    else:
        start_pos, end_pos = _get_correct_alignement(context, answer)
        que_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
    
    que_gen_target = f"{question}"
    return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}

def process_e2e_qg( paragraph):
    source_text = f"generate questions: {paragraph['context'].strip()}"
    questions = [qas['question'].strip() for qas in paragraph['qas']]
    target_text = " {sep_token} ".join(questions)
    target_text = f"{target_text} {{sep_token}}"
    return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

def process_ans_ext( paragraph):
    context = paragraph['context'].strip()

    # split into sentences
    sents = nltk.sent_tokenize(context)

    # get positions of the sentences
    positions = []
    for i, sent in enumerate(sents):
        if i == 0:
            start, end = 0, len(sent)
        else:
            start, end = (prev_end + 1), (prev_end + len(sent) + 1)
        prev_end = end
        positions.append({'start': start, 'end': end})
    
    # get answers
    answers = [qa['answers'][0] for qa in paragraph['qas'] if len(qa["answers"]) > 0]

    # get list of answers for each sentence
    sent_answers = []
    for pos, sent in zip(positions, sents):
        target_answers = []
        for ans in answers:
            if ans['answer_start'] in range(pos['start'], pos['end']):
                target_answers.append(ans['text'].strip())
        sent_answers.append(target_answers)

    # build inputs and targets
    examples = []
    for i, ans in enumerate(sent_answers):
        context = "extract answers:"
        if len(ans) == 0: continue
        ans = list(set(ans))
        for j, sent in enumerate(sents):
            if i == j:
                sent = "{hl_token} %s {hl_token}" % sent
            context = "%s %s" % (context, sent)
            context = context.strip()
        input_text = context
        target_text = " {sep_token} ".join(ans) + " {sep_token}"

        examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})
    
    return examples

def _generate_examples( filepath):
    """This function returns the examples in the raw (text) form."""
    logging.info("generating examples from = %s", filepath)
    count = 0
    tasks = ['e2e_qg']
    outputs = []
    with open(filepath) as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                if type(paragraph["context"])==str:
                    context = paragraph["context"].strip()
                    
                    if 'ans_ext' in tasks:
                        ans_ext_examples = process_ans_ext(paragraph)
                        for example in ans_ext_examples:
                                outputs.append(example)
                                count += 1
                    
                    if 'e2e_qg' in tasks:
                        print(count)
                        outputs.append(process_e2e_qg(paragraph))
                        count += 1
                    
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        for task in tasks:
                            if task == 'qa':
                                if len(answers) > 0:
                                    outputs.append(process_qa_text(context, question, answers[0]))
                                    count += 1
                            
                            if task == 'qg':
                                if len(answers) > 0:
                                    outputs.append(process_qg_text(context, question, qa["answers"][0]))
                                    count += 1
    return outputs

outs = _generate_examples("dev_ensemble.json")
print(len(outs))
# data = {"data":outs}
# with open("train_ensemble_hl.json","w") as f:
#     f.write(json.dumps(data))