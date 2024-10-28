using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace VertoTest.Models
{
    public class ContentPanels
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public required int ContentItemID { get; set; }

        public string? Title { get; set; }
        public required int Order { get; set; }
        public bool IsActive { get; set; }

        public ICollection<StaticContent>? StaticContents { get; set; }

    }
}
