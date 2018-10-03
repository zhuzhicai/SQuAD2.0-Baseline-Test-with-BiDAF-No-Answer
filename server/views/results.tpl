% rebase('main.tpl')
<h2>BiDAF-No-Answer (Single Model)</h2>
<p><b>Paragraph:</b> {{paragraph}}</p>
<p><b>Question:</b> {{question}}</p>
<p><b>Model Predictions:</b>
<ol>
% for phrase, prob in beam:
  % if not phrase:
    % phrase = '<No Answer>'
  % end
  <li> [p={{'%.3g' % prob}}] {{phrase}}
% end
</ol>
</p>
% include('form.tpl', paragraph=paragraph, question=question)
