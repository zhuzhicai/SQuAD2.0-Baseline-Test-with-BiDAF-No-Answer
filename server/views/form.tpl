<form action="/post_query" method="post" id="form">
<p><b>Enter a paragraph:</b><br/>
<textarea rows="3" cols="80" name="paragraph" id="p-area">
{{!paragraph}}
</textarea>
</p>

<p><b>Enter a question:</b><br/>
<textarea rows="2" cols="80" name="question" id="q-area">
{{!question}}
</textarea>
</p>

<input type="submit" class="btn btn-primary"/>
</form>
<script type="text/javascript">
//<![CDATA[
var form = document.getElementById("form");
var p_area = document.getElementById("p-area");
var q_area = document.getElementById("q-area");
p_area.addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.keyCode == 13) {
    this.form.submit();
  }
});
q_area.addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.keyCode == 13) {
    this.form.submit();
  }
});
//]]>
</script>
