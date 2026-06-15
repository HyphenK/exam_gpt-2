# exam_gpt-2
ECO4126-01 Final Project #2

# 사용한 원문과 그에 대한 설명
https://www.gutenberg.org/ebooks/10 (킹 제임스 성경, KJV)

https://www.gutenberg.org/cache/epub/10/pg10.txt (원문 다운로드 링크)

위의 원문에서, 맨 앞 부분의 메타데이터 및 목차 관련 부분을 제거한 후, The First Book of Moses: Called Genesis 부분부터 학습에 사용되었습니다.

원문은 1611년 출판된 성경의 영어 번역본 중 영미권에서 가장 널리 알려진 킹 제임스 성경, 약칭 KJV이며, 고전 영문학에서 수업 당시 사용한 셰익스피어의 희곡들과 비슷한 위상을 가지면서, 본문 내에 현재는 사용되지 않는 옛 영어 표현을 다수 가지고 있는데, 모델이 이러한 고전 영어 표현을 잘 학습하는지 확인이 용이하기에 훈련용으로 선택하였습니다.

# 사용한 코드와 그에 대한 설명
코드는 수업에서 제공된 notebook_06.ipynb 에 약간의 수정사항을 추가하여 제작되었습니다.

## 1. Tokenizer
원본 GPT-2(그리고 이후의 모델들)와 가장 결정적인 차이를 가지는 부분으로, 원본 GPT-2가 단어 단위로 토큰을 생성하는 것과 비교하여, 본 코드는 알파벳과 문장부호 등의 글자 한 개 한 개에 대하여 토큰을 생성합니다.

예시로서,

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
```

이 부분에서 전체 텍스트에 등장하는 문자들을 모아 vocabulary를 만들고, 각 문자를 숫자 ID로 바꾸는데,

```text
"hello"
```

를 대략

```text
['h', 'e', 'l', 'l', 'o']
```

처럼 문자 단위로 보고 토큰을 생성합니다.

## 2. Token Embedding

위의 예시에서,

```text
['h', 'e', 'l', 'l', 'o']
```

는 토큰화를 알파벳 순으로 한다고 가정하였을 때(이는 코드 내에서 실제로 이루어지는 과정과 크게 다르지 않음), 대략 다음과 같이 이루어집니다.

```text
[8, 5, 12, 12, 15]
```

이러한 숫자만으로는 예측을 하기 위한 어떠한 연산을 하기 어렵기 때문에, one-hot encoding을 이용하여 전체 N개(코드에서는 64개)의 수를 갖는 리스트(벡터) 중 8의 경우 8번째 수만 1이고 나머지는 0, 5의 경우 5번째 수만 1이고 나머지는 0, ...의 방법으로 토큰을 변환합니다.

## 3. Position Embedding

앞선 예시의 세 번째 l과 네 번째 l은 그 위치가 다르므로, 벡터에 주어지는 정보 또한 달라야 합니다. 이를 위해 앞선 Token Embedding 과정을 이용해 만든 리스트(벡터)에, 이 글자의 위치에 대한 정보도 추가한 것이 최종적으로 글자를 숫자 형태의 데이터로 변환한 결과가 됩니다.

# GPT 답변 전문 (참고용)
좋아. 처음 질문으로 돌아가면, 네 파일은 **GPT-2의 축소판 구조**를 꽤 잘 담고 있어. 다만 GPT-2 자체는 더 큰 모델이고, tokenizer나 학습 데이터 규모 등은 훨씬 복잡하다. 여기서는 **GPT-2의 구성품이 각각 무슨 역할을 하는지**를 네 코드와 연결해서 설명할게.

---

## 전체 그림

GPT-2는 크게 보면 이런 흐름이야.

```text
입력 문장
→ tokenization
→ token embedding
→ position embedding
→ Transformer block 여러 층
    → masked multi-head self-attention
    → feedforward network
    → residual connection
    → layer normalization
→ final layer norm
→ language modeling head
→ 다음 토큰 확률분포
```

네 파일의 `TinyGPT`도 이 흐름을 거의 그대로 가지고 있다. `token_embedding`, `position_embedding`, 여러 개의 `Block`, 마지막 `LayerNorm`, `lm_head`가 들어 있다 .

---

## 4. Causal masked self-attention

GPT-2의 핵심은 **masked self-attention**이다.

Self-attention은 각 토큰이 문장 안의 다른 토큰들을 참고해서 자신의 표현을 업데이트하는 구조다.

예를 들어:

```text
The cat sat on the mat
```

에서 `"sat"`은 `"cat"`을 참고해야 자연스럽다.

하지만 GPT는 다음 토큰을 예측하는 모델이다. 그러므로 현재 위치에서 **미래 토큰을 보면 안 된다.**

예를 들어 학습 중에:

```text
Genesis 1:1 In the beginning
```

에서 `"In"`을 예측할 때, 뒤에 있는 `"the beginning"`을 미리 보면 cheating이 된다.

그래서 GPT는 causal mask를 사용한다. 네 코드에서는 이 부분이다.

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

그리고 attention score에 mask를 적용한다.

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

이렇게 하면 미래 위치는 attention할 수 없게 된다 .

간단히 말하면:

```text
1번째 토큰 → 1번째만 봄
2번째 토큰 → 1, 2번째만 봄
3번째 토큰 → 1, 2, 3번째만 봄
4번째 토큰 → 1, 2, 3, 4번째만 봄
```

이 구조 때문에 GPT는 왼쪽에서 오른쪽으로 글을 생성할 수 있다.

**역할 요약:**
Causal masked self-attention은 각 토큰이 이전 문맥만 보고 다음 토큰을 예측하게 만든다.

---

## 5. Query, Key, Value

Self-attention 내부에는 Query, Key, Value가 있다.

네 코드에서는:

```python
self.key = nn.Linear(emb_dim, head_size, bias=False)
self.query = nn.Linear(emb_dim, head_size, bias=False)
self.value = nn.Linear(emb_dim, head_size, bias=False)
```

로 정의되어 있다 .

직관적으로는 이렇게 보면 된다.

| 요소    | 역할                   |
| ----- | -------------------- |
| Query | “나는 무엇을 찾고 있는가?”     |
| Key   | “나는 어떤 정보를 가진 토큰인가?” |
| Value | “실제로 전달할 정보는 무엇인가?”  |

예를 들어 어떤 위치의 토큰이 “주어가 무엇인지” 알고 싶다면, 그 토큰의 Query가 이전 토큰들의 Key와 비교된다. 관련성이 높으면 attention weight가 커지고, 그 토큰의 Value를 많이 가져온다.

네 코드에서는 attention score를 이렇게 계산한다.

```python
wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
```

즉, Query와 Key의 유사도를 계산한다 .

그다음 softmax를 통해 확률처럼 만들고:

```python
wei = F.softmax(wei, dim=-1)
```

마지막으로 Value를 가중합한다.

```python
out = wei @ v
```

**역할 요약:**
Query, Key, Value는 각 토큰이 이전 문맥 중 어떤 정보를 얼마나 참고할지 결정하는 장치다.

---

## 6. Multi-head attention

한 개의 attention head만 있으면 한 종류의 관계만 잘 볼 수 있다.

하지만 언어에는 여러 관계가 있다.

예를 들어 한 head는:

```text
주어-동사 관계
```

를 볼 수 있고, 다른 head는:

```text
따옴표 안 대화 구조
```

를 볼 수 있고, 또 다른 head는:

```text
장/절 번호 패턴
```

을 볼 수도 있다.

그래서 GPT-2는 여러 attention head를 병렬로 사용한다. 이게 multi-head attention이다.

네 코드에서는:

```python
self.heads = nn.ModuleList([
    Head(emb_dim, head_size, block_size, dropout)
    for _ in range(num_heads)
])
```

로 여러 head를 만들고, forward에서 결과를 이어 붙인다.

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```

그 후 다시 projection한다.

```python
out = self.proj(out)
```



**역할 요약:**
Multi-head attention은 여러 관점에서 문맥 관계를 동시에 보게 해준다.

---

## 7. FeedForward Network, MLP

Attention은 주로 “토큰들 사이의 관계”를 계산한다.

그 다음에는 각 위치별로 독립적인 변환을 해준다. 이게 feedforward network다.

네 코드에서는:

```python
self.net = nn.Sequential(
    nn.Linear(emb_dim, 4 * emb_dim),
    nn.ReLU(),
    nn.Linear(4 * emb_dim, emb_dim),
    nn.Dropout(dropout),
)
```

로 되어 있다 .

구조는:

```text
128차원 → 512차원 → 128차원
```

처럼 잠깐 차원을 키웠다가 다시 줄인다.

이 부분은 각 토큰 위치의 표현을 더 복잡하게 가공하는 역할을 한다.

Attention이:

```text
어디를 볼 것인가?
```

를 담당한다면, FeedForward는:

```text
그 정보를 어떻게 해석하고 변환할 것인가?
```

를 담당한다.

GPT-2에서는 ReLU 대신 보통 GELU 계열 활성화 함수를 쓴다. 네 코드는 교육용이라 ReLU로 단순화되어 있다.

**역할 요약:**
FeedForward network는 attention으로 모은 정보를 각 위치별로 비선형 변환해서 더 풍부한 표현으로 만든다.

---

## 8. Residual connection

Transformer block에서는 attention이나 feedforward의 결과를 그냥 새 값으로 바꾸지 않는다. 원래 입력에 더한다.

네 코드에서는:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

라고 되어 있다 .

이게 residual connection이다.

직관적으로 말하면:

```text
새로 계산한 정보로 기존 정보를 완전히 덮어쓰지 말고,
기존 정보에 새 정보를 조금 더하자.
```

이다.

이게 중요한 이유는 층이 깊어질수록 학습이 어려워지기 때문이다. residual connection이 있으면 정보와 gradient가 더 쉽게 흐른다.

**역할 요약:**
Residual connection은 기존 정보를 보존하면서 새 정보를 추가하고, 깊은 모델이 안정적으로 학습되도록 돕는다.

---

## 9. Layer Normalization

LayerNorm은 각 token representation의 값 분포를 정규화한다.

네 코드에서는 block 안에:

```python
self.ln1 = nn.LayerNorm(emb_dim)
self.ln2 = nn.LayerNorm(emb_dim)
```

가 있고, 마지막에도:

```python
self.ln_f = nn.LayerNorm(emb_dim)
```

가 있다  .

LayerNorm은 대략 이런 역할이다.

```text
너무 큰 값, 너무 작은 값이 생기지 않게 조절해서
학습을 안정적으로 만든다.
```

네 코드의 block은 pre-layernorm 방식이다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

즉, attention이나 feedforward에 넣기 전에 먼저 LayerNorm을 한다.

**역할 요약:**
LayerNorm은 각 층의 입력 분포를 안정화해서 학습이 잘 되게 만든다.

---

## 10. Transformer Block

GPT-2의 기본 단위는 Transformer block이다.

네 코드에서는 `Block`이 이 역할을 한다.

```python
class Block(nn.Module):
    ...
    self.ln1 = nn.LayerNorm(emb_dim)
    self.sa = MultiHeadAttention(...)
    self.ln2 = nn.LayerNorm(emb_dim)
    self.ffwd = FeedForward(...)
```

그리고 forward는:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

이다 .

즉, 한 block은 다음을 포함한다.

```text
LayerNorm
→ Masked Multi-Head Self-Attention
→ Residual Add
→ LayerNorm
→ FeedForward
→ Residual Add
```

**역할 요약:**
Transformer block은 문맥 정보를 섞고, 각 토큰 표현을 가공하는 GPT의 핵심 반복 단위다.

---

## 11. Block stacking

GPT-2는 Transformer block을 여러 층 쌓는다.

네 코드에서는:

```python
self.blocks = nn.Sequential(*[
    Block(emb_dim, num_heads, block_size, dropout)
    for _ in range(num_layers)
])
```

로 block을 여러 개 쌓는다 .

`num_layers=4`면 block이 4개다.

층이 깊어질수록 모델은 더 복잡한 패턴을 배울 수 있다.

낮은 층은 대체로:

```text
문자/단어 수준 패턴
```

을 배우고, 높은 층은:

```text
문장 구조, 문체, 장기 문맥
```

을 배울 가능성이 커진다.

**역할 요약:**
Block stacking은 단순한 패턴에서 복잡한 문맥 패턴까지 여러 단계로 표현을 정제하게 해준다.

---

## 12. Final LayerNorm

모든 block을 지난 후 마지막으로 한 번 더 LayerNorm을 한다.

네 코드에서는:

```python
h = self.blocks(h)
h = self.ln_f(h)
```

이다 .

이것은 최종 출력 representation을 안정적으로 정리하는 역할을 한다.

**역할 요약:**
Final LayerNorm은 여러 Transformer block을 지난 최종 표현을 안정화한다.

---

## 13. Language Modeling Head

GPT의 목적은 다음 토큰을 맞히는 것이다.

따라서 마지막 hidden vector를 vocabulary 크기의 점수로 바꿔야 한다.

네 코드에서는:

```python
self.lm_head = nn.Linear(emb_dim, vocab_size)
```

그리고:

```python
logits = self.lm_head(h)
```

로 되어 있다 .

출력 `logits`의 shape은 대략:

```text
(batch_size, sequence_length, vocab_size)
```

이다.

각 위치마다 vocabulary 전체에 대한 점수가 나온다.

예를 들어 어떤 위치에서 다음 문자가 될 후보가:

```text
'a', 'b', 'c', ..., '\n'
```

라면, `lm_head`는 각각에 대한 점수를 만든다.

**역할 요약:**
Language modeling head는 최종 hidden vector를 다음 토큰 후보들의 점수로 바꾼다.

---

## 14. Softmax

`lm_head`가 출력하는 것은 확률이 아니라 logit이다.

예를 들어:

```text
a: 2.1
b: -0.4
c: 0.8
```

같은 점수다.

이걸 확률분포로 바꾸려면 softmax를 쓴다.

네 sampling 코드에서는:

```python
probs = F.softmax(logits, dim=-1)
```

로 확률을 만든다 .

**역할 요약:**
Softmax는 vocabulary 전체에 대한 점수를 확률분포로 바꾼다.

---

## 15. Sampling

학습이 끝난 뒤에는 모델이 다음 토큰 확률분포를 낸다.

그중 하나를 뽑고, 다시 문맥 뒤에 붙인다.

네 코드에서는:

```python
ix = torch.multinomial(probs, num_samples=1)
out.append(itos[ix.item()])
context = torch.cat([context[:, 1:], ix], dim=1)
```

로 되어 있다 .

즉:

```text
현재 문맥 → 다음 문자 예측 → 뽑은 문자를 문맥에 추가 → 다시 예측
```

을 반복한다.

이게 GPT가 글을 생성하는 방식이다.

**역할 요약:**
Sampling은 모델의 다음 토큰 확률분포에서 실제로 출력할 토큰을 선택하는 과정이다.

---

## 16. Loss function

학습할 때는 모델의 예측과 정답을 비교해야 한다.

네 코드에서는 cross entropy loss를 사용한다.

```python
def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)
```



Dataset은 입력 `x`와 정답 `y`를 한 칸 차이 나게 만든다.

```python
x = self.data[idx : idx + self.block_size]
y = self.data[idx + 1 : idx + self.block_size + 1]
```



예를 들어 원문이:

```text
hello
```

라면 학습 구조는:

```text
입력: h e l l
정답: e l l o
```

이다.

즉, 매 위치에서 다음 문자를 맞히도록 학습한다.

**역할 요약:**
Cross entropy loss는 모델이 예측한 다음 토큰 확률과 실제 다음 토큰을 비교해서 오차를 계산한다.

---

## 17. Optimizer

loss가 계산되면 모델 파라미터를 업데이트해야 한다.

네 코드에서는 AdamW를 쓴다.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

그리고 학습 loop에서:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

을 수행한다 .

역할은 간단히 말해:

```text
틀린 만큼 가중치를 조금씩 고쳐서 다음에는 더 잘 맞히게 하는 것
```

이다.

**역할 요약:**
Optimizer는 loss를 줄이는 방향으로 모델 파라미터를 업데이트한다.

---

## GPT-2 구성품을 한 줄씩 요약하면

| 구성품                  | 역할                                       |
| -------------------- | ---------------------------------------- |
| Tokenizer            | 문장을 토큰 ID로 변환                            |
| Token embedding      | 토큰 ID를 의미 벡터로 변환                         |
| Position embedding   | 토큰의 위치 정보를 추가                            |
| Causal mask          | 미래 토큰을 보지 못하게 막음                         |
| Self-attention       | 이전 문맥 중 중요한 토큰을 참고                       |
| Multi-head attention | 여러 관점에서 문맥 관계를 분석                        |
| FeedForward / MLP    | 각 위치의 정보를 비선형적으로 가공                      |
| Residual connection  | 기존 정보를 보존하고 학습 안정화                       |
| LayerNorm            | 값 분포를 정규화해 학습 안정화                        |
| Transformer block    | attention + MLP + residual + norm의 기본 단위 |
| Block stacking       | 여러 block을 쌓아 복잡한 패턴 학습                   |
| Final LayerNorm      | 최종 hidden representation 안정화             |
| LM head              | hidden vector를 vocabulary 점수로 변환         |
| Softmax              | 점수를 확률분포로 변환                             |
| Sampling             | 확률분포에서 다음 토큰 선택                          |
| Loss                 | 예측과 정답의 차이 계산                            |
| Optimizer            | 파라미터 업데이트                                |

---

## 네 파일과 GPT-2의 관계

네 파일은 진짜 GPT-2는 아니지만, 핵심 구조는 거의 같다.

네 파일의 제목도 “Toward a Tiny GPT”이고, 추가하는 구성품으로 multi-head attention, feedforward network, residual connection, layer normalization, block stacking을 명시하고 있다 .

즉, 네 파일은:

```text
GPT-2의 원리를 이해하기 위한 교육용 미니어처 구현
```

이라고 보면 된다.

다만 차이는 있다.

| 항목         | 네 파일 TinyGPT     | GPT-2            |
| ---------- | ---------------- | ---------------- |
| 입력 단위      | 문자 단위            | subword token 단위 |
| tokenizer  | 직접 만든 char vocab | BPE tokenizer    |
| 모델 크기      | 매우 작음            | 훨씬 큼             |
| 데이터        | input.txt 하나     | 대규모 웹 텍스트        |
| activation | ReLU             | 주로 GELU          |
| 목적         | 교육용 구조 이해        | 실제 대규모 언어모델      |

핵심적으로는 이렇게 이해하면 된다.

> GPT-2는 “이전 토큰들을 보고 다음 토큰의 확률분포를 예측하는 decoder-only Transformer”이고, 네 파일은 그 구조를 문자 단위로 작게 구현한 버전이다.
